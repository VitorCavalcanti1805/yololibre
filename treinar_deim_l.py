# -*- coding: utf-8 -*-
"""Treina o DEIM-L (DEIM-D-FINE com backbone HGNetv2-B4, ~31M parâmetros) da libreyolo no dataset "data" (COCO).

É o mesmo treino do treinar_deim.py (augmentations com a sombra do YOLOX, métricas por época, best.pt por
--melhor-por, avaliação final em validação e teste, AMP em fp16). Só mudam o tamanho do modelo e alguns padrões:
  - peso inicial: pesos/deim_dfine_hgnetv2_l_coco_50e.pth (DEIM-D-FINE-L oficial treinado no COCO);
  - experimento: runs/train/deim_l_sombra;
  - batch 4 (o L ocupa bem mais memória que o S na RTX 4050 de 6 GB; use --batch 2 se faltar memória).
A libreyolo ajusta sozinha o que depende do tamanho (backbone com lr x0,05, stem e BatchNorm do backbone congelados,
6 camadas no decoder), como no repositório oficial do DEIM.

Exemplos (na raiz do projeto, com o ambiente conda "libreyolo" ativado):
  python treinar_deim_l.py --checar                       # só confere dataset/augmentations, NÃO treina
  python treinar_deim_l.py                                # treina a partir do DEIM-L do COCO
  python treinar_deim_l.py --data C:/Users/voce/Downloads/data --nome l_v1
  python treinar_deim_l.py --nome l_v1 --batch 2          # se faltar memória na GPU
  python treinar_deim_l.py --nome l_v1 --retomar          # continua do last.pt
  python treinar_deim_l.py --nome l_v1 --somente-avaliar runs/train/l_v1/weights/best.pt
Todas as opções do treinar_deim.py valem aqui (python treinar_deim_l.py -h).
"""
import treinar_deim as t

t.TAMANHO = "l"
t.PESOS = "pesos/deim_dfine_hgnetv2_l_coco_50e.pth"
t.NOME = "deim_l_sombra"
t.BATCH = 4
t.DESCRICAO = __doc__

if __name__ == "__main__":  # obrigatório no Windows por causa dos workers do DataLoader
    t.main()
