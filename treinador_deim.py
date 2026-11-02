# -*- coding: utf-8 -*-
"""Trainer e acompanhamento do treino do DEIM, com as mesmas funcionalidades do treinar_yolox.py.

TreinadorDEIM (DEIMTrainer + sombra sintética):
  - em cada validação, além do mAP/mAR da libreyolo, calcula precision/recall/F1 (total e por classe) com o
    limiar de confiança por classe (--conf) e o IoU mínimo (--iou), e AP50/AP50:95/AR100 por classe;
  - essas métricas entram no metrics.jsonl da libreyolo e o best.pt é escolhido por --melhor-por
    (F1 por padrão), sem perder a opção de retomar o treino.

AcompanhamentoTreino (callback da libreyolo):
  - grava historico.csv (uma linha por época: lr, tempo, losses de treino e, nas épocas avaliadas, loss de
    validação, mAP/mAR, P/R/F1 totais e por classe);
  - imprime o resumo por época (tendência da loss, overfitting, melhor modelo) e salva em resumo_epocas.txt;
  - atualiza os gráficos de perdas/métricas por época a cada validação.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

import metricas_deim as md
from aug_yolox_deim import DEIMTrainerComSombra

log = logging.getLogger("treino_deim")

CHAVE_MELHOR = {"f1": "metrics/F1", "map50": "metrics/mAP50", "map50_95": "metrics/mAP50-95"}


class TreinadorDEIM(DEIMTrainerComSombra):
    """DEIMTrainer com sombra + P/R/F1 por classe na validação e best.pt escolhido por --melhor-por."""

    conf: Sequence[float] = (0.5, 0.6)
    iou: float = 0.5
    best_metric_key: str = CHAVE_MELHOR["f1"]   # lido pelo BaseTrainer (best.pt, checkpoint, resume)

    def _run_validation(self, epoch: int, *, save_plots: Optional[bool] = None):
        m = super()._run_validation(epoch, save_plots=save_plots)
        if not m:
            return m
        validador = getattr(self, "_epoch_validator", None)
        try:
            r = md.resultado_do_validador(validador, "valid", self.conf, self.iou, completo=False)
            m["metrics"].update(md.metricas_planas(r))
        except Exception:
            log.exception("Falha ao calcular precision/recall/F1 da validação")
        chave = self.best_metric_key
        m["best_metric"] = float(m["metrics"].get(chave, 0.0))
        m["best_metric_key"] = chave
        return m


def instalar(conf: Sequence[float], iou: float, melhor_por: str) -> None:
    """Faz o LibreDEIM.train() usar o TreinadorDEIM (ele importa o trainer na hora do treino)."""
    import libreyolo.models.deim.trainer as deim_trainer

    TreinadorDEIM.conf = tuple(conf)
    TreinadorDEIM.iou = float(iou)
    TreinadorDEIM.best_metric_key = CHAVE_MELHOR[melhor_por]
    deim_trainer.DEIMTrainer = TreinadorDEIM


# ============================================================
# ACOMPANHAMENTO POR ÉPOCA
# ============================================================

def tendencia(valores: list[float], n: int = 5) -> float:
    """Inclinação relativa (%/época) dos últimos n valores; negativo = caindo."""
    v = np.array(valores[-n:], float)
    if len(v) < 3 or not np.isfinite(v).all():
        return float("nan")
    return float(np.polyfit(np.arange(len(v)), v, 1)[0] / (abs(v.mean()) + 1e-9) * 100)


def _nome_val(k: str) -> str:
    """metrics/mAP50-95 -> val_mAP50_95; metrics/val_loss_mal -> val_loss_mal."""
    k = k.split("/", 1)[-1].replace("-", "_").replace("(B)", "").replace("/", "_")
    return k if k.startswith("val_") else f"val_{k}"


class AcompanhamentoTreino:
    """Callback da libreyolo: historico.csv, resumo por época e gráficos de convergência."""

    def __init__(self, nomes: list[str], conf: Sequence[float], iou: float, melhor_por: str) -> None:
        self.nomes, self.conf, self.iou, self.melhor_por = list(nomes), list(conf), iou, melhor_por
        self.historico: list[dict[str, float]] = []
        self.pasta: Optional[Path] = None
        self.total = 0
        self.inicio_sem_aug = 0
        self.avaliacoes_sem_melhora = 0

    # --- eventos da libreyolo ---
    def on_train_start(self, ev: Any) -> None:
        from libreyolo.data.augment.detr import resolve_aug_stop_epoch

        self.pasta, self.total = Path(ev.save_dir), int(ev.total_epochs)
        cfg = dict(ev.config)
        self.inicio_sem_aug = resolve_aug_stop_epoch(self.total, cfg.get("aug_stop_epoch_ratio", 1.0),
                                                     cfg.get("no_aug_epochs", 0)) + 1
        # ao retomar, mantém as épocas já treinadas do historico.csv
        self.historico = [{**h, "epoca": int(h["epoca"])} for h in md.ler_csv(self.pasta / "historico.csv")
                          if h.get("epoca", 0) < ev.start_epoch]
        self._log(f"Classes: {self.nomes} | saída {self.pasta.resolve()} | épocas {ev.start_epoch}..{self.total} | "
                  f"aug forte até a época {self.inicio_sem_aug - 1} | best.pt por {self.melhor_por} | "
                  f"P/R/F1 com conf {md._txt_conf(self.conf)} e IoU {self.iou:.2f}")

    def on_train_epoch_end(self, ev: Any) -> None:
        lr = max(ev.lr.values()) if ev.lr else float("nan")
        linha: dict[str, float] = {"epoca": ev.epoch, "lr": lr, "tempo_s": ev.epoch_seconds,
                                   "sem_aug": float(ev.epoch >= self.inicio_sem_aug), "treino_loss": ev.train_loss}
        linha.update({f"treino_{k}": float(v) for k, v in ev.train_loss_items.items()})
        if ev.validated:
            linha.update({_nome_val(k): float(v) for k, v in ev.val_metrics.items()})
            # a loss total de validação é a soma das parcelas (o DEIM reporta só as parcelas)
            parcelas = [v for k, v in linha.items() if k.startswith("val_") and "loss" in k and k != "val_loss"]
            if parcelas and "val_loss" not in linha:
                linha["val_loss"] = float(sum(parcelas))
            self.avaliacoes_sem_melhora = 0 if ev.is_best else self.avaliacoes_sem_melhora + 1
            linha["melhor"] = float(ev.is_best)
        self.historico.append(linha)
        md.gravar_csv(self.historico, self.pasta / "historico.csv")
        self._resumo(linha, ev)
        if ev.validated:
            try:
                md.plotar_historico(self.historico, self.pasta / "graficos", self.nomes, self.conf, self.iou)
            except Exception:
                log.exception("Falha ao plotar o histórico")

    def on_train_end(self, ev: Any) -> None:
        md.plotar_historico(self.historico, self.pasta / "graficos", self.nomes, self.conf, self.iou)

    # --- saída ---
    def _log(self, txt: str) -> None:
        print(txt, flush=True)
        if self.pasta is not None:
            with open(self.pasta / "resumo_epocas.txt", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {txt}\n")

    def _resumo(self, l: dict[str, float], ev: Any) -> None:
        loss_tr = [h["treino_loss"] for h in self.historico]
        delta = loss_tr[-1] - loss_tr[-2] if len(loss_tr) > 1 else float("nan")
        incl = tendencia(loss_tr)
        if math.isnan(incl):
            estado = "aquecendo"
        elif incl < -1.0:
            estado = "caindo (convergindo)"
        elif incl < 0.3:
            estado = "estável (platô)"
        else:
            estado = "SUBINDO (instável; reduza o lr?)"
        comps = "  ".join(f"{k[7:]} {v:.3f}" for k, v in l.items() if k.startswith("treino_") and k != "treino_loss")
        txt = (f"\n[época {l['epoca']:.0f}/{self.total}] {l['tempo_s']:.0f}s  lr {l['lr']:.2e}"
               f"{'  (sem aug forte)' if l['sem_aug'] else ''}\n"
               f"  loss treino {loss_tr[-1]:.3f} (delta {delta:+.3f}; últimas 5: {incl:+.2f}%/época -> {estado})  {comps}")
        if ev.validated:
            av = [h for h in self.historico if "val_mAP50" in h]
            loss_val = [h["val_loss"] for h in av if "val_loss" in h]
            txt += (f"\n  validação: loss {l.get('val_loss', float('nan')):.3f}  mAP50 {l.get('val_mAP50', 0):.3f}  "
                    f"mAP50:95 {l.get('val_mAP50_95', 0):.3f}  mAR100 {l.get('val_AR100', 0):.3f}  |  "
                    f"conf {md._txt_conf(self.conf)}, IoU {self.iou:.2f}: P {l.get('val_P', 0):.3f}  "
                    f"R {l.get('val_R', 0):.3f}  F1 {l.get('val_F1', 0):.3f}")
            por_cls = "  ".join(f"{n}: F1 {l.get(f'val_F1_{n}', 0):.3f} AP50 {l.get(f'val_AP50_{n}', 0):.3f}"
                                for n in self.nomes)
            txt += f"\n  por classe: {por_cls}"
            if ev.best_metric is not None:
                txt += f"\n  melhor {self.melhor_por} = {ev.best_metric:.4f} na época {ev.best_epoch}"
                txt += " <- NOVO MELHOR (best.pt salvo)" if ev.is_best else \
                    f" (sem melhora há {self.avaliacoes_sem_melhora} avaliações)"
            if len(loss_val) >= 3 and loss_val[-1] > min(loss_val) * 1.05 and incl < 0:
                txt += "\n  ATENÇÃO: loss de validação subindo enquanto a de treino cai -> sinal de overfitting"
        self._log(txt)
