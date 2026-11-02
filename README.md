# yololibre: treino do DEIM-S para panelas e remendos

Treina o **DEIM-S** (DEIM-D-FINE, cerca de 10M de parâmetros) da biblioteca [libreyolo](https://pypi.org/project/libreyolo/) para detectar as classes `panela` e `remendo`, com as mesmas augmentations e métricas usadas no treino do YOLOX.

- **Augmentations:** as do próprio DEIM (distorção fotométrica, zoom out, recorte por IoU, flip horizontal e multi-escala por batch) mais a sombra sintética do YOLOX (postes, árvores, folhagem e veículos), aplicada com probabilidade 0,5 só no treino.
- **Durante o treino:** avalia na validação a cada N épocas (mAP/mAR, loss de validação e precision/recall/F1 total e por classe), grava o histórico e atualiza os gráficos. O `best.pt` é escolhido pelo F1.
- **No fim:** avalia o `best.pt` na validação e no teste e gera curvas, matriz de confusão e um relatório.

## Arquivos

| arquivo | o que faz |
|---|---|
| `treinar_deim.py` | script principal: é ele que você roda |
| `treinador_deim.py` | trainer do DEIM com as métricas extras e o acompanhamento por época |
| `metricas_deim.py` | cálculo de métricas, gráficos e relatório |
| `aug_yolox_deim.py` | sombra sintética do YOLOX integrada ao DEIM |
| `data.yaml` | configuração do dataset (aponta para a pasta `data/`) |
| `pesos/deim_dfine.pth` | peso inicial: DEIM-D-FINE-S oficial treinado no COCO |

## 1. Instalar o ambiente

Com o Anaconda (no Windows, use o **Anaconda Prompt** ou o terminal do PyCharm):

```powershell
conda create -n libreyolo python=3.12 -y
conda activate libreyolo
```

Instale o PyTorch com suporte à GPU. O índice depende da "CUDA Version" que aparece no `nvidia-smi`:

| CUDA Version no `nvidia-smi` | índice |
|---|---|
| 12.6 ou 12.7 | `cu126` |
| 12.8 ou mais | `cu128` |

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install libreyolo matplotlib
```

Confira se a GPU foi reconhecida (a saída deve terminar em `True`):

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Versões testadas: Python 3.12, torch 2.14.1+cu126, libreyolo 1.7.1 e matplotlib 3.11, numa RTX 4050 de 6 GB.

No PyCharm, aponte o interpretador para esse ambiente: **Settings → Python Interpreter → Add Interpreter → Conda Environment → Use existing → `libreyolo`**.

## 2. Preparar o dataset

O dataset não vai para o repositório. Coloque a pasta `data` na raiz do projeto, ao lado do `data.yaml`, no formato COCO:

```
yololibre/
├── data.yaml
└── data/
    ├── annotations/
    │   ├── train.json
    │   ├── valid.json
    │   └── test.json
    ├── train/images/
    ├── valid/images/
    └── test/images/
```

As classes vêm do `data.yaml`: `0: panela` e `1: remendo`. Elas precisam bater com os `category_id` dos JSONs.

**Outro dataset:** você não precisa editar o `data.yaml`. Passe a pasta pelo terminal com `--data`, e o script descobre sozinho os splits e as classes a partir dos JSONs:

```powershell
python treinar_deim.py --data C:/caminho/outro_dataset --nome outro_v1
```

O `--data` aceita os dois layouts COCO mais comuns: o de cima (`annotations/*.json` + `<split>/images/`) e o do Roboflow (`<split>/_annotations.coco.json`). O split de validação pode se chamar `valid` ou `val`. O de teste é opcional.

## 3. Conferir antes de treinar

```powershell
python treinar_deim.py --checar
```

Esse comando não treina. Ele monta o dataset de treino com as augmentations, testa o DataLoader e salva 8 amostras com as caixas desenhadas em `previas_augmentation/`. Abra as imagens e veja se as caixas caem em cima dos defeitos.

## 4. Treinar

Partindo do peso do COCO que está no repositório (recomendado):

```powershell
python treinar_deim.py --nome s_v1 --pesos pesos/deim_dfine.pth
```

Outras opções de ponto de partida:

```powershell
python treinar_deim.py --nome s_v1                                                  # baixa o DEIM-S do COCO (LibreDEIMs.pt)
python treinar_deim.py --nome s_v2 --pesos runs/train/s_v1/weights/best.pt          # continua a partir de um modelo seu
python treinar_deim.py --nome s_v1 --pesos nenhum                                   # do zero (não recomendado)
```

Quando o peso foi treinado com outro número de classes (o COCO tem 80), a cabeça do modelo é refeita sozinha para as 2 classes.

O padrão é 132 épocas, batch 8 e imagem 640. **Se der erro de memória da GPU** ("CUDA out of memory"), baixe o batch:

```powershell
python treinar_deim.py --nome s_v1 --pesos pesos/deim_dfine.pth --batch 4
```

### Continuar um treino interrompido

Se o treino parou no meio (queda de energia, Ctrl+C), retome do último checkpoint com o mesmo `--nome`:

```powershell
python treinar_deim.py --nome s_v1 --retomar
```

Ele continua da época seguinte, com o mesmo total de épocas e o mesmo learning rate do treino original. Para retomar de um checkpoint específico, passe o caminho dele: `--retomar runs/train/s_v1/weights/epoch_50.pt`. Esses checkpoints por época só existem se o treino rodou com `--salvar-a-cada N`.

## 5. Acompanhar o treino

A cada época, o terminal mostra um resumo como este:

```
[época 18/132] 14s  lr 4.00e-04
  loss treino 25.191 (delta +0.981; últimas 5: -1.39%/época -> caindo (convergindo))  mal 10.668  bbox 2.612 ...
  validação: loss 17.195  mAP50 0.592  mAP50:95 0.324  mAR100 0.640  |  conf 0.50 / 0.60, IoU 0.50: P 0.667  R 0.600  F1 0.632
  por classe: panela: F1 0.667 AP50 0.634  remendo: F1 0.600 AP50 0.551
  melhor f1 = 0.6316 na época 18 <- NOVO MELHOR (best.pt salvo)
```

O resumo também avisa quando a loss de validação sobe enquanto a de treino cai, o que é sinal de overfitting.

Tudo fica salvo em `runs/train/<nome>/`:

| arquivo | conteúdo |
|---|---|
| `weights/best.pt` | melhor modelo na validação (pelo F1, ou pelo critério de `--melhor-por`) |
| `weights/last.pt` | modelo da última época, usado pelo `--retomar` |
| `historico.csv` | uma linha por época: lr, tempo, losses de treino e, nas épocas avaliadas, as métricas de validação |
| `resumo_epocas.txt` | os resumos impressos no terminal |
| `graficos/perdas.png` | losses de treino e validação e learning rate por época |
| `graficos/metricas_por_epoca.png` | mAP, mAR, P/R/F1 e métricas por classe ao longo do treino |
| `metrics.jsonl`, `train.log` | logs da própria libreyolo |

Os dois gráficos são atualizados a cada validação, então dá para abri-los com o treino ainda rodando.

## 6. Resultado final

No fim do treino, o `best.pt` é avaliado na validação e no teste. São gerados:

| arquivo | conteúdo |
|---|---|
| `resumo.md` | tabela com mAP50, mAP50:95, mAR, precision, recall e F1 por split e por classe |
| `metricas_finais.json` | as mesmas métricas, mais a matriz de confusão, em JSON |
| `graficos/pr_<split>.png` | curva precision × recall (IoU 0,50) |
| `graficos/confianca_<split>.png` | precision, recall e F1 em função do limiar de confiança, com o limiar de melhor F1 |
| `graficos/confusao_<split>.png` | matriz de confusão panela × remendo × fundo |
| `graficos/barras_<split>.png` | métricas por classe em barras |

Para avaliar de novo um modelo já treinado (por exemplo, com outros limiares), sem treinar:

```powershell
python treinar_deim.py --nome s_v1 --somente-avaliar runs/train/s_v1/weights/best.pt --conf 0.4 0.5
```

## Opções

Lista completa: `python treinar_deim.py --help`.

| opção | padrão | o que faz |
|---|---|---|
| `--data` | `data.yaml` | dataset: um `.yaml` ou a pasta do dataset |
| `--nome` | `deim_s_sombra` | nome do experimento (subpasta em `runs/train/`) |
| `--projeto` | `runs/train` | pasta onde ficam os experimentos |
| `--pesos` | `LibreDEIMs.pt` | peso inicial: `pesos/deim_dfine.pth`, um `.pt` seu ou `nenhum` |
| `--retomar` | | continua um treino interrompido (do `last.pt`, ou do caminho passado) |
| `--epocas` | 132 | total de épocas |
| `--batch` | 8 | imagens por batch (baixe para 4 se faltar memória) |
| `--imgsz` | 640 | tamanho da imagem |
| `--workers` | 4 | processos que carregam as imagens |
| `--eval-intervalo` | 5 | avalia na validação a cada N épocas (e sempre na última) |
| `--paciencia` | 0 | para o treino se o `best.pt` não melhorar em N épocas (0 = nunca para) |
| `--salvar-a-cada` | 0 | guarda `weights/epoch_N.pt` a cada N épocas (0 = não guarda) |
| `--sem-loss-val` | | não calcula a loss de validação (um pouco mais rápido) |
| `--sombra` | 0.5 | probabilidade da sombra sintética (0 desliga) |
| `--seed` | 0 | semente aleatória |
| `--conf` | `0.5 0.6` | limiar de confiança por classe, na ordem das classes (panela, remendo), usado em P/R/F1 e na matriz de confusão |
| `--iou` | 0.5 | IoU mínimo para uma detecção contar como acerto em P/R/F1 |
| `--melhor-por` | `f1` | métrica que escolhe o `best.pt`: `f1`, `map50` ou `map50_95` |
| `--sem-teste` | | não avalia o split de teste no fim |
| `--somente-avaliar` | | pula o treino e só avalia o checkpoint passado |
| `--checar` | | só confere o dataset e as augmentations, sem treinar |

## Sobre o peso `pesos/deim_dfine.pth`

É o checkpoint oficial do DEIM-D-FINE-S treinado no COCO, do repositório [DEIM](https://github.com/ShihuaHuang95/DEIM) (licença Apache 2.0). Como ele não está no formato da libreyolo, o script o carrega direto como DEIM-S. Ele deve ser o mesmo modelo que a libreyolo baixa como `LibreDEIMs.pt`; tê-lo no repositório só evita o download.
