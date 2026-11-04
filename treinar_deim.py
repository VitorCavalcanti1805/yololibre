# -*- coding: utf-8 -*-
"""Treina o DEIM-S (DEIM-D-FINE, ~10.2M parâmetros) da libreyolo no dataset "data" (COCO).

Augmentations: as do próprio DEIM + sombra sintética do YOLOX (ver aug_yolox_deim.py).
Durante o treino (como no treinar_yolox.py): avaliação na validação a cada --eval-intervalo épocas com
mAP/mAR + precision/recall/F1 por classe, loss de validação, resumo de convergência por época, historico.csv
e best.pt escolhido por --melhor-por. No fim, o best.pt é avaliado em validação e teste, com gráficos.

Saída em <projeto>/<nome>/:
  weights/best.pt, weights/last.pt           modelos (best = melhor --melhor-por na validação)
  historico.csv, resumo_epocas.txt           uma linha por época / resumo impresso a cada época
  metrics.jsonl                               log padrão da libreyolo (inclui P/R/F1 por classe)
  metricas_finais.json, resumo.md             métricas do best.pt em valid e test
  graficos/*.png                              perdas, métricas por época, curvas PR, P/R/F1 x confiança,
                                              matriz de confusão, barras por classe

Exemplos (na raiz do projeto, com o ambiente conda "libreyolo" ativado):
  python treinar_deim.py --checar             # só confere dataset/augmentations e salva prévias, NÃO treina
  python treinar_deim.py                      # treina a partir do DEIM-S do COCO (LibreDEIMs.pt)
  python treinar_deim.py --data C:/Users/voce/Downloads/data --nome s_v1   # escolhe o dataset (pasta ou .yaml)
  python treinar_deim.py --nome s_v2 --pesos runs/train/s_v1/weights/best.pt   # parte de um modelo seu
  python treinar_deim.py --nome s_v1 --pesos nenhum                            # do zero
  python treinar_deim.py --nome s_v1 --retomar                                 # continua do last.pt
  python treinar_deim.py --nome s_v1 --somente-avaliar runs/train/s_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

DATA_YAML = "data.yaml"
# O tamanho do modelo e os padrões abaixo são trocados por treinar_deim_l.py (DEIM-L)
TAMANHO = "s"            # n, s, m, l ou x
PESOS = "LibreDEIMs.pt"  # DEIM-S pré-treinado no COCO (baixado automaticamente)
NOME = "deim_s_sombra"
BATCH = 8                # RTX 4050 6 GB; baixe para 4 se faltar memória
EPOCAS = 132             # padrão do DEIM
DESCRICAO = __doc__


def rotulo() -> str:
    return f"DEIM-{TAMANHO.upper()}"


def args():
    p = argparse.ArgumentParser(description=DESCRICAO, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("dados e saída")
    g.add_argument("--data", default=DATA_YAML,
                   help="dataset: a PASTA do dataset COCO (ex.: C:/Users/voce/Downloads/data) ou um .yaml "
                        f"(padrão: {DATA_YAML}). Com uma pasta, o yaml é gerado sozinho a partir dos JSONs")
    g.add_argument("--nome", default=NOME, help="subpasta do experimento em --projeto")
    g.add_argument("--projeto", default="runs/train")
    g.add_argument("--sem-teste", action="store_true", help="não avalia o split de teste no fim")
    g.add_argument("--splits", nargs="+", choices=["val", "test"], default=None,
                   help="splits da avaliação final (padrão: val e test). Ex.: --splits test")

    g = p.add_argument_group("treino")
    g.add_argument("--pesos", default=PESOS,
                   help=f"checkpoint inicial: '{PESOS}' = {rotulo()} do COCO (padrão), 'nenhum' = do zero, ou o caminho "
                        "de um .pt seu (ex.: runs/train/s_v1/weights/best.pt); a cabeça é refeita se o nº de "
                        "classes não bater")
    g.add_argument("--retomar", nargs="?", const="auto", default=None,
                   help="continua um treino interrompido: sem valor usa <projeto>/<nome>/weights/last.pt; "
                        "ou passe o caminho do last.pt (ignora --pesos)")
    g.add_argument("--epocas", type=int, default=None,
                   help=f"total de épocas (padrão {EPOCAS}, o do DEIM; com --retomar, o do treino original)")
    g.add_argument("--batch", type=int, default=BATCH, help=f"padrão {BATCH} (RTX 4050 6 GB)")
    g.add_argument("--imgsz", type=int, default=640)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--eval-intervalo", type=int, default=5, help="avalia na validação a cada N épocas (e na última)")
    g.add_argument("--paciencia", type=int, default=0,
                   help="para se --melhor-por não melhorar em N épocas (0 = nunca)")
    g.add_argument("--salvar-a-cada", type=int, default=0, help="guarda weights/epoch_N.pt a cada N épocas (0 = não)")
    g.add_argument("--sem-loss-val", action="store_true", help="não calcula a loss na validação (um pouco mais rápido)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--sombra", type=float, default=0.5, help="probabilidade da sombra sintética")

    g = p.add_argument_group("métricas")
    g.add_argument("--conf", type=float, nargs="+", default=[0.5, 0.6],
                   help="limiar de confiança por classe (na ordem do category_id) para precision/recall/F1")
    g.add_argument("--iou", type=float, default=0.5, help="IoU mínimo para uma detecção contar como acerto em P/R/F1")
    g.add_argument("--melhor-por", choices=["f1", "map50", "map50_95"], default="f1", help="métrica do best.pt")
    g.add_argument("--somente-avaliar", default=None, help="pula o treino e avalia/plota esse checkpoint")
    g.add_argument("--checar", action="store_true", help="só confere o pipeline e salva prévias, sem treinar")
    return p.parse_args()


SPLITS = {"train": ("train",), "val": ("valid", "val", "validation"), "test": ("test",)}


def _achar_split(raiz: Path, nomes: tuple[str, ...]) -> tuple[str, str] | None:
    """(pasta de imagens, json) relativos à raiz, nos dois layouts COCO mais comuns:
    annotations/<split>.json + <split>/images/   ou   <split>/_annotations.coco.json (Roboflow)."""
    for n in nomes:
        for js in (raiz / "annotations" / f"{n}.json", raiz / n / "_annotations.coco.json"):
            if js.exists():
                img = raiz / n / "images" if (raiz / n / "images").is_dir() else raiz / n
                if img.is_dir():
                    return img.relative_to(raiz).as_posix(), js.relative_to(raiz).as_posix()
    return None


def yaml_com_caminho_absoluto(arquivo: Path) -> str:
    """A libreyolo procura um 'path' relativo em ~/datasets; aqui ele passa a ser relativo à pasta do yaml
    (ex.: path: data -> <projeto>/data). Se mudar algo, grava a cópia em configs_dataset/<nome>.yaml."""
    import yaml

    cfg = yaml.safe_load(arquivo.read_text(encoding="utf-8")) or {}
    caminho = cfg.get("path")
    if not caminho or Path(caminho).is_absolute():
        return str(arquivo)
    raiz = (arquivo.resolve().parent / caminho).resolve()
    if not raiz.is_dir():
        raise FileNotFoundError(f"{arquivo}: a pasta do dataset '{caminho}' não existe em {raiz}")
    cfg["path"] = raiz.as_posix()
    saida = Path("configs_dataset") / arquivo.name
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(f"# cópia de {arquivo.resolve()} com o path absoluto\n"
                     + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Dataset: {raiz}")
    return str(saida)


def resolver_dataset(data: str) -> str:
    """Aceita um .yaml (usa como está) ou a pasta do dataset (gera configs_dataset/<pasta>.yaml)."""
    import json

    p = Path(data).expanduser()
    if p.is_file():
        return yaml_com_caminho_absoluto(p)
    if not p.is_dir():
        raise FileNotFoundError(f"--data {data}: não existe (passe a pasta do dataset ou um .yaml)")
    raiz = p.resolve()
    achados = {s: _achar_split(raiz, n) for s, n in SPLITS.items()}
    for s in ("train", "val"):
        if achados[s] is None:
            raise FileNotFoundError(
                f"--data {raiz}: não achei o split '{s}'. Esperado annotations/{SPLITS[s][0]}.json + "
                f"{SPLITS[s][0]}/images/ ou {SPLITS[s][0]}/_annotations.coco.json")
    cats = sorted(json.load(open(raiz / achados["train"][1], encoding="utf-8"))["categories"], key=lambda c: c["id"])
    linhas = [f"# gerado por treinar_deim.py a partir de {raiz}", f"path: {raiz.as_posix()}"]
    linhas += [f"{s}: {v[0]}" for s, v in achados.items() if v]
    linhas += ["", "annotations:"] + [f"  {s}: {v[1]}" for s, v in achados.items() if v]
    linhas += ["", f"nc: {len(cats)}", "names:"] + [f"  {i}: {json.dumps(str(c['name']), ensure_ascii=False)}"
                                                   for i, c in enumerate(cats)]
    saida = Path("configs_dataset") / f"{raiz.name}.yaml"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"Dataset {raiz}: {', '.join(s for s, v in achados.items() if v)} | classes "
          f"{[c['name'] for c in cats]} | config em {saida}")
    if achados["test"] is None:
        print("  (sem split de teste: a avaliação final fica só na validação)")
    return str(saida)


def nomes_classes(data: str) -> list[str]:
    from libreyolo.data import load_data_config

    nomes = load_data_config(data)["names"]
    return [nomes[k] for k in sorted(nomes)] if isinstance(nomes, dict) else list(nomes)


def checar(a):
    """Monta o mesmo dataset de treino que o trainer usaria e salva algumas amostras com as caixas."""
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader

    import aug_yolox_deim as aug
    from libreyolo import LibreDEIM
    from libreyolo.data import get_coco_annotation_file, get_coco_image_dir, load_data_config
    from libreyolo.data.dataset import COCODataset
    from libreyolo.models.deim.transforms import DEIMMultiScaleCollate, DEIMTrainTransform

    aug.SHADOW_PROB = a.sombra
    m = LibreDEIM(None, size=TAMANHO, nb_classes=len(nomes_classes(a.data)), device="cpu")
    print(f"{rotulo()}: {sum(p.numel() for p in m.model.parameters()) / 1e6:.2f} M parametros")

    cfg = load_data_config(a.data)
    preproc = DEIMTrainTransform(max_labels=120, flip_prob=0.5, imgsz=a.imgsz)
    base = COCODataset(
        data_dir=cfg["root"],
        json_file=get_coco_annotation_file(cfg, "train"),
        name=get_coco_image_dir(cfg, "train", "train"),
        img_size=(a.imgsz, a.imgsz),
        preproc=preproc,
        num_classes=int(cfg["nc"]),
        names=cfg["names"],
    )
    ds = aug.DEIMComSombraDataset(base, (a.imgsz, a.imgsz), preproc=preproc)
    print(f"treino: {len(ds)} imagens | sombra p={aug.SHADOW_PROB}")

    # passa pelo DataLoader com workers para garantir que funciona no Windows (spawn)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=min(2, a.workers),
                    collate_fn=DEIMMultiScaleCollate(base_size=a.imgsz))
    imgs, labels, _, _ = next(iter(dl))
    print(f"batch ok: imagens {tuple(imgs.shape)} | alvos {tuple(labels.shape)}")

    saida = Path("previas_augmentation")
    saida.mkdir(exist_ok=True)
    nomes = cfg["names"]
    for i in range(8):
        img, lab, _, _ = ds[random_idx(len(ds))]
        vis = (img.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
        for c, cx, cy, w, h in lab[lab[:, 3] > 0]:
            x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0) if int(c) == 0 else (0, 128, 255), 2)
            cv2.putText(vis, str(nomes[int(c)]), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(saida / f"amostra_{i}.jpg"), vis)
    print(f"previas salvas em {saida.resolve()}")


def random_idx(n):
    import random

    return random.randrange(n)


def carregar_modelo_inicial(a):
    """--retomar > --pesos <arquivo> > --pesos nenhum (modelo do zero)."""
    from libreyolo import LibreDEIM, LibreYOLO

    if a.retomar:
        ckpt = Path(a.projeto) / a.nome / "weights" / "last.pt" if a.retomar == "auto" else Path(a.retomar)
        if not ckpt.exists():
            raise FileNotFoundError(f"--retomar: {ckpt} não existe")
        print(f"Retomando de {ckpt}")
        return LibreYOLO(str(ckpt)), str(ckpt)
    if str(a.pesos).lower() in ("nenhum", "none", ""):
        print(f"Sem pesos iniciais: {rotulo()} treinado do zero")
        return LibreDEIM(None, size=TAMANHO, nb_classes=len(nomes_classes(a.data))), None
    existe = Path(a.pesos).exists()
    if not existe and not a.pesos.startswith("LibreDEIM"):  # LibreDEIM*.pt a libreyolo baixa sozinha
        raise FileNotFoundError(f"--pesos {a.pesos} não existe")
    print(f"Checkpoint inicial: {a.pesos}")
    if existe and not eh_checkpoint_libreyolo(a.pesos):
        # peso oficial do repositório DEIM (ex.: deim_dfine_hgnetv2_l_coco_50e.pth): a libreyolo não sabe pelo
        # nome se é DEIM ou D-FINE (mesma arquitetura), então carrega direto como DEIM no tamanho deste script
        print(f"  peso fora do formato da libreyolo (repositório oficial do DEIM): carregando como {rotulo()}")
        return LibreDEIM(a.pesos, size=TAMANHO), None
    return LibreYOLO(a.pesos), None  # a cabeça é refeita para as classes do data.yaml se precisar


def eh_checkpoint_libreyolo(caminho: str) -> bool:
    """True se o arquivo tem os metadados da libreyolo (model_family etc.)."""
    import torch

    try:
        ck = torch.load(caminho, map_location="cpu", weights_only=False)
    except Exception:
        return True  # deixa a libreyolo dar a mensagem de erro
    return isinstance(ck, dict) and "model_family" in ck


def avaliar_final(a, ckpt: Path, pasta: Path) -> None:
    """Avalia o checkpoint em validação e teste; gera gráficos, metricas_finais.json e resumo.md."""
    import metricas_deim as md
    from libreyolo.data import load_data_config

    cfg = load_data_config(a.data)
    splits = a.splits or (["val"] + (["test"] if not a.sem_teste else []))
    if "test" in splits and not cfg.get("test"):
        print("Aviso: o dataset não tem split de teste; avaliando só o que existe")
        splits = [s for s in splits if s != "test"] or ["val"]
    resultados = []
    for s in splits:
        print(f"Avaliando {ckpt} em {s}...")
        r = md.avaliar_checkpoint(ckpt, a.data, s, a.conf, a.iou, imgsz=a.imgsz, batch=a.batch,
                                  workers=a.workers, salvar_em=pasta / f"avaliacao_{s}")
        md.plotar_avaliacao(r, pasta / "graficos", a.conf, a.iou)
        resultados.append(r)
    print("\n" + md.relatorio(resultados, ckpt, pasta, a.data, a.conf, a.iou))
    print(f"Modelos, métricas e gráficos em {pasta.resolve()}")


def epocas_do_checkpoint(caminho: str) -> int:
    """Total de épocas do treino que gerou o last.pt (o agendamento do lr depende dele)."""
    import torch

    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    return int((ck.get("config") or {}).get("epochs") or EPOCAS)


def treinar(a):
    import aug_yolox_deim as aug
    import treinador_deim as td

    aug.SHADOW_PROB = a.sombra
    td.instalar(a.conf, a.iou, a.melhor_por)
    acompanhamento = td.AcompanhamentoTreino(nomes_classes(a.data), a.conf, a.iou, a.melhor_por)

    model, retomar = carregar_modelo_inicial(a)
    if a.epocas is None:
        a.epocas = epocas_do_checkpoint(retomar) if retomar else EPOCAS
    print(f"Total de épocas: {a.epocas}")
    resultados = model.train(
        data=a.data,
        epochs=a.epocas,
        batch=a.batch,
        imgsz=a.imgsz,
        workers=a.workers,
        project=a.projeto,
        name=a.nome,
        exist_ok=bool(retomar),       # ao retomar, continua gravando na mesma pasta
        resume=retomar or False,
        amp=True,
        amp_dtype="float16",          # AMP em fp16 (treino e validações durante o treino)
        seed=a.seed,
        patience=a.paciencia,
        eval_interval=a.eval_intervalo,
        save_period=a.salvar_a_cada,  # 0 = não guarda epoch_N.pt
        val_loss=not a.sem_loss_val,
        callbacks=[acompanhamento],
    )
    pasta = Path(resultados["save_dir"])
    best = resultados.get("best_checkpoint") or resultados.get("last_checkpoint")
    print(f"Melhor época: {resultados.get('best_epoch')} | checkpoint: {best}")
    if best:
        avaliar_final(a, Path(best), pasta)


def somente_avaliar(a):
    import metricas_deim as md

    pasta = Path(a.projeto) / a.nome
    pasta.mkdir(parents=True, exist_ok=True)
    hist = md.ler_csv(pasta / "historico.csv")
    if hist:
        md.plotar_historico(hist, pasta / "graficos", nomes_classes(a.data), a.conf, a.iou)
    avaliar_final(a, Path(a.somente_avaliar), pasta)


def main():
    a = args()
    a.data = resolver_dataset(a.data)
    if a.checar:
        checar(a)
    elif a.somente_avaliar:
        somente_avaliar(a)
    else:
        treinar(a)


if __name__ == "__main__":  # obrigatório no Windows por causa dos workers do DataLoader
    main()
