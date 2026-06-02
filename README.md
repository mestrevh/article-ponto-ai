# Pipeline Experimental — Artigo ERBASE 2026

> **Avaliação Comparativa de Modelos de Extração de Embeddings Faciais para Reconhecimento em Vídeo**

Este módulo contém todo o código e dados do experimento apresentado no artigo submetido ao **ERBASE — Encontro Nacional de Inteligência Artificial e Computacional**, edição 2026.

---

## Objetivo

Avaliar e comparar o desempenho de **7 modelos de extração de embeddings faciais** em um cenário realista de reconhecimento de pessoas em vídeo, utilizando um pipeline end-to-end que combina detecção (YOLO), tracking (DeepSort), extração de embeddings e busca vetorial (ChromaDB) com sistema de votação.

---

## Arquitetura do Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Pipeline Experimental                            │
│                                                                         │
│  1. Popular ChromaDB         Imagens de rosto → Embeddings → ChromaDB   │
│         │                                                               │
│  2. Detecção (YOLO)          Extrair rostos de cada frame do vídeo      │
│         │                                                               │
│  3. Tracking (DeepSort)      Associar rostos ao longo dos frames        │
│         │                                                               │
│  4. Extração de Embedding    Modelo sob teste gera vetor por frame      │
│         │                                                               │
│  5. Busca Vetorial           Comparar embedding com ChromaDB (cosine)   │
│         │                                                               │
│  6. Sistema de Votação       Acumular votos por track → vencedor        │
│         │                                                               │
│  7. Cálculo de Métricas      Accuracy, ROC/AUC, TAR@FAR → CSV           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Modelos Avaliados

| Modelo | Provider | Dimensão | Backbone | Descrição |
|---|---|---|---|---|
| **FaceNet** | DeepFace | 128-D | InceptionResNet | Google FaceNet (triplet loss) |
| **ArcFace** | DeepFace | 512-D | ResNet-100 | Additive angular margin (CVPR 2019) |
| **CosFace** | ONNX | 512-D | iResNet-100 | Large margin cosine loss (CVPR 2018) |
| **SphereFace** | ONNX | 512-D | iResNet-100 | Angular softmax / A-Softmax (CVPR 2017) |
| **MagFace** | ONNX | 512-D | iResNet-100 | Magnitude-aware angular margin (CVPR 2021) |
| **CurricularFace** | ONNX | 512-D | iResNet-100 | Curriculum learning angular margin (CVPR 2020) |
| **ElasticFace** | ONNX | 512-D | iResNet-100 | Elastic margin loss (CVPR-W 2022) |

> Os modelos ONNX são baixados e convertidos automaticamente via `download_models.py` a partir dos checkpoints oficiais dos autores.

---

## Métricas Calculadas

### Qualidade do Embedding
- **Accuracy** — Acurácia do sistema de votação por track
- **ROC Curve (AUC)** — Área sob a curva ROC frame a frame
- **TAR @ FAR 1%** — True Acceptance Rate com False Acceptance Rate fixo em 1%

### Similaridade Vetorial
- **Recall@K** — Proporção de matches corretos nos K vizinhos mais próximos
- **Precision@K** — Precisão nos K vizinhos mais próximos

Os resultados são exportados em arquivos CSV individuais por modelo (ex: `arcface.csv`, `facenet.csv`).
