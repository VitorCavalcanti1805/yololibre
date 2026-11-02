# -*- coding: utf-8 -*-
"""Sombra sintética do YOLOX (exps/example/custom/yolox_s_custom.py) acrescentada ao DEIM da libreyolo.

As augmentations do próprio DEIM continuam iguais (sem mosaic/mixup, que o DEIM não usa):
  RandomPhotometricDistort (p=0.5), RandomZoomOut, RandomIoUCrop (p=0.8), flip horizontal (0.5),
  multi-scale por batch (480..800). As fortes desligam no início das no_aug_epochs.

O que entra a mais: a sombra realista do YOLOX, com shadow_prob = 0.5, aplicada na imagem original
antes das augmentations do DEIM. Só no treino; validação e teste ficam sem sombra.
"""
from __future__ import annotations

import random

import cv2
import numpy as np

from libreyolo.models.deim.trainer import DEIMTrainer
from libreyolo.models.deim.transforms import DEIMPassThroughDataset

# Mesmo valor do yolox_s_custom.py
SHADOW_PROB = 0.5


# ============================================================
# SOMBRAS SINTÉTICAS REALISTAS (cópia do yolox_s_custom.py)
# ============================================================

def _forma_poste(h, w):
    """Faixa fina e comprida (poste, placa, fio)."""
    m = np.zeros((h, w), np.uint8)
    largura = random.randint(max(2, w // 60), max(3, w // 15))
    x0 = random.randint(0, w)
    x1 = x0 + random.randint(-w // 2, w // 2)
    y0 = random.randint(int(h * 0.3), h)
    y1 = random.randint(int(h * 0.3), h)
    cv2.line(m, (x0, y0), (x1, y1 if abs(y1 - y0) > h // 6 else h), 255, largura)
    return m


def _forma_arvore(h, w):
    """Mancha recortada formada por vários blobs (copa de árvore)."""
    m = np.zeros((h, w), np.uint8)
    cx = random.randint(0, w)
    cy = random.randint(int(h * 0.35), h)
    raio = random.randint(w // 20, w // 6)
    for _ in range(random.randint(3, 9)):
        ex = cx + random.randint(-raio, raio)
        ey = cy + random.randint(-raio // 2, raio // 2)
        cv2.ellipse(m, (ex, ey), (random.randint(raio // 3, raio), random.randint(raio // 4, raio // 2)),
                    random.randint(0, 180), 0, 360, 255, -1)
    return m


def _forma_folhagem(h, w):
    """Sombra manchada (luz passando entre folhas)."""
    ruido = np.random.rand(max(2, h // 24), max(2, w // 24)).astype(np.float32)
    ruido = cv2.resize(ruido, (w, h), interpolation=cv2.INTER_CUBIC)
    m = (ruido > random.uniform(0.5, 0.7)).astype(np.uint8) * 255
    regiao = np.zeros((h, w), np.uint8)
    x0 = random.randint(0, w // 2)
    y0 = random.randint(int(h * 0.3), int(h * 0.7))
    cv2.rectangle(regiao, (x0, y0), (x0 + random.randint(w // 4, w // 2), h), 255, -1)
    return cv2.bitwise_and(m, regiao)


def _forma_veiculo(h, w):
    """Quadrilátero (sombra de carro, caminhão ou da própria viatura)."""
    m = np.zeros((h, w), np.uint8)
    cx = random.randint(0, w)
    cy = random.randint(int(h * 0.45), h)
    bw = random.randint(w // 8, w // 3)
    bh = random.randint(h // 12, h // 5)
    pts = np.array([[cx - bw // 2 + random.randint(-bw // 4, bw // 4), cy - bh // 2],
                    [cx + bw // 2 + random.randint(-bw // 4, bw // 4), cy - bh // 2],
                    [cx + bw // 2, cy + bh // 2],
                    [cx - bw // 2, cy + bh // 2]], np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m


FORMAS = [_forma_poste, _forma_arvore, _forma_folhagem, _forma_veiculo]


def sombra_realista(img):
    """Aplica de 1 a 3 sombras de formatos variados. Não altera a imagem original."""
    h, w = img.shape[:2]
    luz = np.ones((h, w), np.float32)
    for _ in range(random.randint(1, 3)):
        forma = random.choice(FORMAS)(h, w).astype(np.float32) / 255.0
        borda = random.choice([1, 3, 5, 9, 15])  # 1 = sombra nítida, maior = sombra suave
        if borda > 1:
            forma = cv2.GaussianBlur(forma, (borda, borda), 0)
        fator = random.uniform(0.45, 0.75)  # quanto da luz sobra na sombra
        luz = np.minimum(luz, 1.0 - (1.0 - fator) * forma)
    out = img.astype(np.float32) * luz[..., None]
    out[..., 0] *= 1.0 + 0.06 * (1.0 - luz)  # sombra real puxa levemente para o azul (BGR)
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# INTEGRAÇÃO COM O DEIM
# ============================================================

class _DatasetComSombra:
    """Envolve o COCODataset da libreyolo e aplica a sombra no pull_item (só no treino).

    Todo o resto (load_anno, annotations, ...) é repassado para o dataset original.
    """

    def __init__(self, dataset, shadow_prob):
        self._ds = dataset
        self._shadow_prob = shadow_prob

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, nome):
        # Protege o unpickle nos workers do DataLoader (o Windows usa spawn)
        if nome.startswith("__") or nome in ("_ds", "_shadow_prob"):
            raise AttributeError(nome)
        return getattr(self._ds, nome)

    def pull_item(self, index):
        item = self._ds.pull_item(index)
        if self._shadow_prob > 0 and random.random() < self._shadow_prob:
            item = (sombra_realista(item[0]),) + tuple(item[1:])
        return item


class DEIMComSombraDataset(DEIMPassThroughDataset):
    """Wrapper padrão do DEIM (sem mosaic) com a sombra aplicada antes das transforms do DEIM."""

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(_DatasetComSombra(dataset, SHADOW_PROB), *args, **kwargs)


class DEIMTrainerComSombra(DEIMTrainer):
    """DEIMTrainer original; só troca o wrapper do dataset de treino."""

    def create_transforms(self):
        preproc, _ = super().create_transforms()
        return preproc, DEIMComSombraDataset


def instalar():
    """Faz o LibreDEIM.train() usar o DEIMTrainerComSombra (ele importa o trainer na hora do treino)."""
    import libreyolo.models.deim.trainer as deim_trainer

    deim_trainer.DEIMTrainer = DEIMTrainerComSombra
