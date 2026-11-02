# -*- coding: utf-8 -*-
"""Métricas, gráficos e relatório do DEIM, iguais aos do treinar_yolox.py.

Parte das detecções que o DetectionValidator da libreyolo já acumula (coco_evaluator.results, formato COCO,
coordenadas da imagem original) e calcula, como no YOLOX:
  - mAP/mAR do COCOeval (geral e por classe) e a curva precision x recall (IoU 0,50);
  - precision/recall/F1 por classe com limiar de confiança por classe (--conf) e IoU mínimo (--iou);
  - curvas P/R/F1 x confiança e o limiar de melhor F1;
  - matriz de confusão classe x classe x fundo.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


# ============================================================
# ESTRUTURAS
# ============================================================

@dataclass
class Predicoes:
    """Detecções de uma imagem em coordenadas originais (xyxy) e o category_id do COCO."""
    caixas: np.ndarray
    scores: np.ndarray
    categorias: np.ndarray


@dataclass
class Curvas:
    """Precision/recall/F1 em função do limiar de confiança, para uma classe."""
    confs: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    tp: np.ndarray       # acertos com score >= conf
    n_det: np.ndarray    # detecções com score >= conf
    n_gt: int


@dataclass
class ResultadoAvaliacao:
    split: str
    coco: dict[str, float]                       # mAP/mAR do COCOeval (todas as classes)
    por_classe: dict[str, dict[str, float]]      # AP50, AP50:95, AR100, P, R, F1, melhor F1 e conf
    total: dict[str, float]                      # P, R, F1 somando as classes nos limiares de --conf
    curvas: dict[str, Curvas] = field(default_factory=dict)
    pr_coco: dict[str, np.ndarray] = field(default_factory=dict)  # precision x recall (IoU 0,50) do COCOeval
    confusao: Optional[np.ndarray] = None        # (C+1)x(C+1): linhas = verdade, colunas = predito; último = fundo
    loss: dict[str, float] = field(default_factory=dict)


def _vazio() -> Predicoes:
    return Predicoes(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))


def predicoes_de_resultados(resultados: Sequence[dict]) -> dict[int, Predicoes]:
    """Lista COCO (image_id, category_id, bbox xywh, score) -> img_id -> Predicoes (xyxy)."""
    por_img: dict[int, list[dict]] = {}
    for r in resultados:
        por_img.setdefault(int(r["image_id"]), []).append(r)
    saida: dict[int, Predicoes] = {}
    for img_id, rs in por_img.items():
        b = np.array([r["bbox"] for r in rs], float).reshape(-1, 4)
        b[:, 2:] += b[:, :2]
        saida[img_id] = Predicoes(b, np.array([r["score"] for r in rs], float),
                                  np.array([r["category_id"] for r in rs], int))
    return saida


# ============================================================
# MÉTRICAS (mesmo cálculo do treinar_yolox.py)
# ============================================================

def iou_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def gt_por_imagem(coco: Any, img_ids: Optional[Sequence[int]] = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """img_id -> (caixas xyxy, category_id) das anotações de verdade."""
    gt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for img_id in (img_ids if img_ids is not None else coco.getImgIds()):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False))
        cx = np.array([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)], float).reshape(-1, 4)
        gt[int(img_id)] = (cx, np.array([a["category_id"] for a in anns], int))
    return gt


def casar(gt: np.ndarray, det: np.ndarray, iou_min: float) -> np.ndarray:
    """Casamento guloso por score (det já ordenado): True para detecção que acerta uma caixa ainda livre."""
    acerto = np.zeros(len(det), bool)
    if not len(gt) or not len(det):
        return acerto
    iou = iou_matriz(det, gt)
    livre = np.ones(len(gt), bool)
    for k in range(len(det)):
        v = np.where(livre, iou[k], -1)
        j = int(v.argmax())
        if v[j] >= iou_min:
            livre[j] = False; acerto[k] = True
    return acerto


def curvas_confianca(gt: dict[int, tuple[np.ndarray, np.ndarray]], preds: dict[int, Predicoes], cat: int,
                     iou_min: float) -> Curvas:
    """P/R/F1 para todos os limiares de confiança de uma vez (o casamento guloso de uma detecção só depende
    das detecções de score maior, então cortar a lista ordenada em qualquer limiar dá o mesmo resultado)."""
    scores, acertos, n_gt = [], [], 0
    for img_id, (gc, gk) in gt.items():
        g = gc[gk == cat]; n_gt += len(g)
        p = preds.get(img_id)
        if p is None:
            continue
        m = p.categorias == cat
        ordem = np.argsort(-p.scores[m])
        s, d = p.scores[m][ordem], p.caixas[m][ordem]
        scores.append(s); acertos.append(casar(g, d, iou_min))
    s = np.concatenate(scores) if scores else np.zeros(0)
    t = np.concatenate(acertos) if acertos else np.zeros(0, bool)
    ordem = np.argsort(-s); s, tp = s[ordem], np.cumsum(t[ordem])
    confs = np.linspace(0, 1, 101)
    k = np.searchsorted(-s, -confs, side="right")   # nº de detecções com score >= conf
    tp_k = np.where(k > 0, tp[np.maximum(k - 1, 0)] if len(tp) else 0, 0).astype(float)
    precision = np.where(k > 0, tp_k / np.maximum(k, 1), 1.0)
    recall = tp_k / max(n_gt, 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    return Curvas(confs, precision, recall, f1, tp_k, k.astype(float), n_gt)


def contagem_no_limiar(c: Curvas, conf: float) -> tuple[float, float, float]:
    """tp, fp e n_gt no limiar (para somar as classes)."""
    i = int(round(conf * 100))
    return float(c.tp[i]), float(c.n_det[i] - c.tp[i]), float(c.n_gt)


def prf(tp: float, fp: float, n: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n if n else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def matriz_confusao(gt: dict[int, tuple[np.ndarray, np.ndarray]], preds: dict[int, Predicoes], cats: list[int],
                    limiares: dict[int, float], iou_min: float) -> np.ndarray:
    """Linhas = classe verdadeira (+ fundo), colunas = classe predita (+ fundo = não detectado)."""
    n = len(cats); M = np.zeros((n + 1, n + 1), int); idx = {c: i for i, c in enumerate(cats)}
    for img_id, (gc, gk) in gt.items():
        p = preds.get(img_id) or _vazio()
        m = np.array([s >= limiares.get(int(c), 1.1) for s, c in zip(p.scores, p.categorias)], bool).reshape(-1)
        ordem = np.argsort(-p.scores[m]); d, dk = p.caixas[m][ordem], p.categorias[m][ordem]
        livre = np.ones(len(gc), bool)
        iou = iou_matriz(d, gc) if len(d) and len(gc) else np.zeros((len(d), len(gc)))
        for k in range(len(d)):
            v = np.where(livre, iou[k], -1) if len(gc) else np.zeros(0)
            if len(v) and v.max() >= iou_min:
                j = int(v.argmax()); livre[j] = False; M[idx[int(gk[j])], idx[int(dk[k])]] += 1
            else:
                M[n, idx[int(dk[k])]] += 1   # falso positivo (fundo detectado como defeito)
        for j in np.where(livre)[0]:
            M[idx[int(gk[j])], n] += 1       # defeito não detectado
    return M


CHAVES_COCO = ["mAP50_95", "mAP50", "mAP75", "mAP_peq", "mAP_med", "mAP_grande",
               "mAR1", "mAR10", "mAR100", "mAR_peq", "mAR_med", "mAR_grande"]


def metricas_coco(coco: Any, preds: dict[int, Predicoes], cats: list[int], nomes: list[str], img_ids: list[int]
                  ) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    from pycocotools.cocoeval import COCOeval
    dets = [{"image_id": i, "category_id": int(c), "score": float(s),
             "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]}
            for i, p in preds.items() for b, s, c in zip(p.caixas, p.scores, p.categorias)]
    if not dets:
        return {k: 0.0 for k in CHAVES_COCO}, {n: {"AP50_95": 0.0, "AP50": 0.0, "AR100": 0.0} for n in nomes}, {}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(dets)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = sorted(img_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    geral = {k: max(float(v), 0.0) for k, v in zip(CHAVES_COCO, ev.stats)}
    prec, rec = ev.eval["precision"], ev.eval["recall"]   # [T, R, K, A, M] e [T, K, A, M]
    media = lambda x: float(x[x > -1].mean()) if (x > -1).any() else 0.0
    k_cat = {c: k for k, c in enumerate(ev.params.catIds)}
    por_classe, pr = {}, {}
    for c, n in zip(cats, nomes):
        k = k_cat[c]
        por_classe[n] = {"AP50_95": media(prec[:, :, k, 0, 2]), "AP50": media(prec[0, :, k, 0, 2]),
                         "AR100": media(rec[:, k, 0, 2])}
        pr[n] = np.clip(prec[0, :, k, 0, 2], 0, None)
    return geral, por_classe, pr


def avaliar_predicoes(coco: Any, resultados: Sequence[dict], split: str, conf: Sequence[float], iou: float,
                      completo: bool = True, img_ids: Optional[Sequence[int]] = None,
                      loss: Optional[dict[str, float]] = None) -> ResultadoAvaliacao:
    """Todas as métricas do YOLOX a partir das detecções COCO (conf baixo, ex.: 0.001) e do GT."""
    cats = sorted(int(c) for c in coco.getCatIds())
    nomes = [str(coco.cats[c]["name"]) for c in cats]
    img_ids = sorted(int(i) for i in (img_ids if img_ids is not None else coco.getImgIds()))
    limiares = {c: float(conf[min(i, len(conf) - 1)]) for i, c in enumerate(cats)}
    preds = predicoes_de_resultados(resultados)
    geral, por_classe, pr = metricas_coco(coco, preds, cats, nomes, img_ids)
    gt = gt_por_imagem(coco, img_ids)
    curvas = {n: curvas_confianca(gt, preds, c, iou) for c, n in zip(cats, nomes)}
    soma = np.zeros(3)
    for c, n in zip(cats, nomes):
        cv = curvas[n]; tp, fp, ng = contagem_no_limiar(cv, limiares[c]); soma += (tp, fp, ng)
        p, r, f = prf(tp, fp, ng); i = int(cv.f1.argmax())
        por_classe[n].update({"conf": limiares[c], "P": p, "R": r, "F1": f, "TP": tp, "FP": fp, "n_gt": ng,
                              "melhor_F1": float(cv.f1[i]), "conf_melhor_F1": float(cv.confs[i])})
    p, r, f = prf(*soma)
    total = {"P": p, "R": r, "F1": f, "TP": float(soma[0]), "FP": float(soma[1]), "n_gt": float(soma[2])}
    res = ResultadoAvaliacao(split, geral, por_classe, total, loss=dict(loss or {}))
    if completo:
        res.curvas, res.pr_coco = curvas, pr
        res.confusao = matriz_confusao(gt, preds, cats, limiares, iou)
    return res


def resultado_do_validador(validador: Any, split: str, conf: Sequence[float], iou: float,
                           completo: bool = True) -> ResultadoAvaliacao:
    """Lê as detecções que o DetectionValidator da libreyolo acabou de acumular e calcula as métricas."""
    ev = validador.coco_evaluator
    img_ids = sorted(getattr(ev, "_img_ids", None) or ev.coco_gt.getImgIds())
    resultados = [{k: r[k] for k in ("image_id", "category_id", "bbox", "score")} for r in ev.results]
    loss = {}
    if hasattr(validador, "_validation_loss_metrics"):
        loss = {k.replace("metrics/", ""): float(v) for k, v in validador._validation_loss_metrics().items()}
    return avaliar_predicoes(ev.coco_gt, resultados, split, conf, iou, completo, img_ids, loss)


def metricas_planas(r: ResultadoAvaliacao) -> dict[str, float]:
    """Métricas extras com prefixo metrics/ para entrar no metrics.jsonl da libreyolo e na escolha do best.pt."""
    m = {f"metrics/{k}": float(r.total[k]) for k in ("P", "R", "F1")}
    for n, d in r.por_classe.items():
        m.update({f"metrics/{k}_{n}": float(d[k]) for k in ("AP50", "AP50_95", "AR100", "P", "R", "F1")})
    return m


def avaliar_checkpoint(ckpt: str | Path, data: str, split: str, conf: Sequence[float], iou: float,
                       imgsz: int = 640, batch: int = 8, workers: int = 4, salvar_em: Optional[Path] = None
                       ) -> ResultadoAvaliacao:
    """Avalia um checkpoint num split do data.yaml ('val' ou 'test') com o validador da libreyolo."""
    import torch
    from libreyolo import LibreYOLO
    from libreyolo.validation import DetectionValidator, ValidationConfig

    modelo = LibreYOLO(str(ckpt))
    cuda = torch.cuda.is_available()
    cfg = ValidationConfig(data=data, split=split, batch_size=batch, imgsz=imgsz, conf_thres=0.001, iou_thres=0.65,
                           max_det=300, device="cuda" if cuda else "cpu", half=cuda, verbose=False,
                           num_workers=workers, save_dir=str(salvar_em) if salvar_em else None)
    v = DetectionValidator(model=modelo, config=cfg)
    v.run()
    return resultado_do_validador(v, "valid" if split == "val" else split, conf, iou, completo=True)


# ============================================================
# HISTÓRICO (CSV)
# ============================================================

def gravar_csv(linhas: list[dict[str, float]], arquivo: Path) -> None:
    chaves = list(dict.fromkeys(k for l in linhas for k in l))
    with open(arquivo, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=chaves); w.writeheader(); w.writerows(linhas)


def ler_csv(arquivo: Path) -> list[dict[str, float]]:
    if not arquivo.exists():
        return []
    with open(arquivo, encoding="utf-8") as f:
        return [{k: float(v) for k, v in l.items() if v not in ("", None)} for l in csv.DictReader(f)]


# ============================================================
# GRÁFICOS
# ============================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _txt_conf(conf: Sequence[float]) -> str:
    return " / ".join(f"{c:.2f}" for c in conf)


def plotar_historico(hist: list[dict[str, float]], pasta: Path, nomes: list[str], conf: Sequence[float],
                     iou: float) -> None:
    """perdas.png (treino x validação, lr) e metricas_por_epoca.png (mAP, mAR, P/R/F1, por classe)."""
    if not hist:
        return
    plt = _plt()
    pasta.mkdir(parents=True, exist_ok=True)
    ep = [h["epoca"] for h in hist]
    av = [h for h in hist if "val_mAP50" in h]; ep_av = [h["epoca"] for h in av]
    inicio_sem_aug = next((h["epoca"] for h in hist if h.get("sem_aug")), None)

    def marcar(ax: Any) -> None:
        if inicio_sem_aug:
            ax.axvline(inicio_sem_aug, color="gray", ls=":", lw=1, label="início sem aug forte")
        ax.grid(alpha=0.3); ax.set_xlabel("época"); ax.legend(fontsize=8)

    # componentes de loss do DEIM (MAL, bbox L1, GIoU, FGL, DDF) e a validação correspondente
    comps = [k[len("treino_"):] for k in hist[-1] if k.startswith("treino_") and k != "treino_loss"]
    val_de = lambda c: next((k for k in (av[-1] if av else {}) if k.startswith("val_") and "loss" in k
                             and k.endswith(c)), None)
    n = 2 + len(comps)
    cols = 3; linhas = -(-n // cols)
    fig, ax = plt.subplots(linhas, cols, figsize=(16, 4.3 * linhas))
    ax = np.atleast_1d(ax).ravel()
    ax[0].plot(ep, [h["treino_loss"] for h in hist], label="treino")
    if av and "val_loss" in av[-1]:
        ax[0].plot(ep_av, [h.get("val_loss", np.nan) for h in av], "o-", ms=3, label="validação")
    ax[0].set_title("loss total")
    for i, c in enumerate(comps, 1):
        ax[i].plot(ep, [h.get(f"treino_{c}", np.nan) for h in hist], label="treino")
        kv = val_de(c)
        if kv:
            ax[i].plot(ep_av, [h.get(kv, np.nan) for h in av], "o-", ms=3, label="validação")
        ax[i].set_title(f"loss {c}")
    ax[n - 1].plot(ep, [h["lr"] for h in hist], label="lr"); ax[n - 1].set_title("learning rate")
    for x in ax[:n]:
        marcar(x)
    for x in ax[n:]:
        x.axis("off")
    fig.suptitle("Convergência do treino"); fig.tight_layout(); fig.savefig(pasta / "perdas.png", dpi=130); plt.close(fig)

    if not av:
        return
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax = ax.ravel()
    for k in ("mAP50", "mAP50_95", "mAP75"):
        ax[0].plot(ep_av, [h.get(f"val_{k}", np.nan) for h in av], "o-", ms=3, label=k.replace("_", ":"))
    ax[0].set_title("mAP (validação)")
    for k in ("AR1", "AR10", "AR100"):
        ax[1].plot(ep_av, [h.get(f"val_{k}", np.nan) for h in av], "o-", ms=3, label="m" + k)
    ax[1].set_title("mAR (validação)")
    for k in ("P", "R", "F1"):
        ax[2].plot(ep_av, [h.get(f"val_{k}", np.nan) for h in av], "o-", ms=3,
                   label={"P": "precision", "R": "recall"}.get(k, k))
    ax[2].set_title(f"precision / recall / F1 (conf {_txt_conf(conf)}, IoU {iou:.2f})")
    for nm in nomes:
        ax[3].plot(ep_av, [h.get(f"val_F1_{nm}", np.nan) for h in av], "o-", ms=3, label=f"F1 {nm}")
        ax[3].plot(ep_av, [h.get(f"val_AP50_{nm}", np.nan) for h in av], "s--", ms=3, label=f"AP50 {nm}")
    ax[3].set_title("por classe (validação)")
    for x in ax:
        marcar(x); x.set_ylim(0, 1)
    fig.suptitle("Métricas de validação por época"); fig.tight_layout()
    fig.savefig(pasta / "metricas_por_epoca.png", dpi=130); plt.close(fig)


def plotar_avaliacao(r: ResultadoAvaliacao, pasta: Path, conf: Sequence[float], iou: float) -> None:
    """pr_<split>.png, confianca_<split>.png, confusao_<split>.png, barras_<split>.png."""
    plt = _plt()
    pasta.mkdir(parents=True, exist_ok=True)
    nomes = list(r.por_classe)
    cores = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # curva precision x recall (COCOeval, IoU 0,50)
    fig, ax = plt.subplots(figsize=(7, 6))
    rec = np.linspace(0, 1, 101)
    for i, n in enumerate(nomes):
        if n in r.pr_coco:
            ax.plot(rec, r.pr_coco[n], color=cores[i], label=f"{n} (AP50 {r.por_classe[n]['AP50']:.3f})")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Precision x recall, IoU 0,50 ({r.split}) - mAP50 {r.coco['mAP50']:.3f}")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(pasta / f"pr_{r.split}.png", dpi=130); plt.close(fig)

    # P, R e F1 em função da confiança
    if r.curvas:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        for j, (k, t) in enumerate([("precision", "precision"), ("recall", "recall"), ("f1", "F1")]):
            for i, n in enumerate(nomes):
                c = r.curvas[n]
                ax[j].plot(c.confs, getattr(c, k), color=cores[i], label=n)
                ax[j].axvline(r.por_classe[n]["conf"], color=cores[i], ls=":", lw=1)
                if k == "f1":
                    ax[j].plot(r.por_classe[n]["conf_melhor_F1"], r.por_classe[n]["melhor_F1"], "o", color=cores[i])
            ax[j].set(xlabel="limiar de confiança", title=f"{t} x confiança (IoU {iou:.2f})", xlim=(0, 1), ylim=(0, 1.02))
            ax[j].grid(alpha=0.3); ax[j].legend()
        fig.suptitle(f"{r.split}: linhas pontilhadas = limiares usados; ponto = melhor F1")
        fig.tight_layout(); fig.savefig(pasta / f"confianca_{r.split}.png", dpi=130); plt.close(fig)

    # matriz de confusão
    if r.confusao is not None:
        rot = nomes + ["fundo"]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(r.confusao, cmap="Blues")
        for (i, j), v in np.ndenumerate(r.confusao):
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > r.confusao.max() / 2 else "black")
        ax.set(xticks=range(len(rot)), yticks=range(len(rot)), xticklabels=rot, yticklabels=rot,
               xlabel="predito", ylabel="verdadeiro",
               title=f"Matriz de confusão ({r.split})\nconf {_txt_conf(conf)}, IoU {iou:.2f}")
        fig.colorbar(im); fig.tight_layout(); fig.savefig(pasta / f"confusao_{r.split}.png", dpi=130); plt.close(fig)

    # barras por classe
    met = ["AP50", "AP50_95", "AR100", "P", "R", "F1"]
    fig, ax = plt.subplots(figsize=(11, 5))
    larg = 0.8 / (len(nomes) + 1)
    for i, n in enumerate(nomes + ["todas"]):
        if n == "todas":
            v = [r.coco["mAP50"], r.coco["mAP50_95"], r.coco["mAR100"], r.total["P"], r.total["R"], r.total["F1"]]
        else:
            v = [r.por_classe[n][m] for m in met]
        b = ax.bar(np.arange(len(met)) + i * larg, v, larg, label=n, color=cores[i % len(cores)])
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(np.arange(len(met)) + larg * len(nomes) / 2, ["AP50", "AP50:95", "AR100", "precision", "recall", "F1"])
    ax.set(ylim=(0, 1.05), title=f"Métricas por classe ({r.split}); P/R/F1 com conf {_txt_conf(conf)} e IoU {iou:.2f}")
    ax.grid(alpha=0.3, axis="y"); ax.legend(); fig.tight_layout()
    fig.savefig(pasta / f"barras_{r.split}.png", dpi=130); plt.close(fig)


# ============================================================
# RELATÓRIO
# ============================================================

def relatorio(resultados: list[ResultadoAvaliacao], ckpt: Path, pasta: Path, data: str, conf: Sequence[float],
              iou: float) -> str:
    """Grava metricas_finais.json e resumo.md; devolve o texto do resumo."""
    pasta.mkdir(parents=True, exist_ok=True)
    with open(pasta / "metricas_finais.json", "w", encoding="utf-8") as f:
        json.dump({r.split: {"coco": r.coco, "total": r.total, "por_classe": r.por_classe, "loss": r.loss,
                             "confusao": r.confusao.tolist() if r.confusao is not None else None} for r in resultados}
                  | {"checkpoint": str(ckpt), "conf": list(conf), "iou": iou}, f, indent=2)
    md = [f"# Avaliação de `{ckpt}`\n", f"Dataset: `{data}`. Precision/recall/F1 com limiar de confiança "
          f"{_txt_conf(conf)} (por classe) e IoU ≥ {iou:.2f}.\n",
          "| split | mAP50 | mAP50:95 | mAP75 | mAR1 | mAR10 | mAR100 | precision | recall | F1 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in resultados:
        c, t = r.coco, r.total
        md.append(f"| {r.split} | {c['mAP50']:.3f} | {c['mAP50_95']:.3f} | {c['mAP75']:.3f} | {c['mAR1']:.3f} | "
                  f"{c['mAR10']:.3f} | {c['mAR100']:.3f} | {t['P']:.3f} | {t['R']:.3f} | {t['F1']:.3f} |")
    md += ["\n| split | classe | AP50 | AP50:95 | AR100 | limiar | precision | recall | F1 | melhor F1 (limiar) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in resultados:
        for n, d in r.por_classe.items():
            md.append(f"| {r.split} | {n} | {d['AP50']:.3f} | {d['AP50_95']:.3f} | {d['AR100']:.3f} | {d['conf']:.2f} "
                      f"| {d['P']:.3f} | {d['R']:.3f} | {d['F1']:.3f} | {d['melhor_F1']:.3f} ({d['conf_melhor_F1']:.2f}) |")
    texto = "\n".join(md) + "\n"
    (pasta / "resumo.md").write_text(texto, encoding="utf-8")
    return texto
