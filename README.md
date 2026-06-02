# Pipeline Experimental — Artigo ERBASE 2026

> **Avaliação Comparativa de Modelos de Extração de Embeddings Faciais para Reconhecimento em Vídeo**

Este repositório contém o pipeline modular e a suíte experimental do artigo submetido ao **ERBASE — Encontro Regional de Computação da Bahia, Alagoas e Sergipe**, edição 2026.

---

## Objetivo

Avaliar e comparar o desempenho de **7 modelos de extração de embeddings faciais** em um cenário realista de processamento e reconhecimento de pessoas em vídeo, utilizando um pipeline modular end-to-end:

1. **Detecção Facial (YOLO):** Localiza bounding boxes de rostos em tempo real.
2. **Rastreamento (DeepSort):** Mantém IDs persistentes (tracks) associando rostos ao longo dos frames.
3. **Pré-processamento:** Aplica redimensionamento uniforme e filtro CLAHE para correção de iluminação.
4. **Extração de Embeddings:** Backbones geram vetores de características L2-normalizados.
5. **Busca Vetorial (ChromaDB):** Compara características contra galeria de fotos de treino por similaridade de cosseno.
6. **Sistema de Votação por Rastro:** Classificação final consolidada por maioria majoritária de votos por `track_id`.

---

## Modelos Avaliados

| Modelo | Provider | Dimensão | Backbone | Descrição |
|---|---|---|---|---|
| **FaceNet** | DeepFace | 512-D | InceptionResNet | Google FaceNet (triplet loss) |
| **ArcFace** | DeepFace | 512-D | ResNet-100 | Additive angular margin (CVPR 2019) |
| **CosFace** | ONNX | 512-D | iResNet-100 | Large margin cosine loss (CVPR 2018) |
| **SphereFace** | ONNX | 512-D | iResNet-100 | Angular softmax / A-Softmax (CVPR 2017) |
| **MagFace** | ONNX | 512-D | iResNet-100 | Magnitude-aware angular margin (CVPR 2021) |
| **CurricularFace** | ONNX | 512-D | iResNet-100 | Curriculum learning angular margin (CVPR 2020) |
| **ElasticFace** | ONNX | 512-D | iResNet-100 | Elastic margin loss (CVPR-W 2022) |

---

## Estrutura do Repositório

```
data/
├── training_faces/         # Galeria de fotos de treino por pessoa (ex: pessoa_1/)
├── recordings/             # Gravações em vídeo para os testes do benchmark
├── models/                 # Modelos ONNX localmente armazenados
└── results/                # Relatórios em CSV contendo métricas avaliadas
src/
├── core/
│   ├── base.py             # Interface base abstrata FaceRecognizer
│   ├── database.py         # Conexão e população do VectorDatabase (ChromaDB)
│   ├── detector.py         # Fluxo de detecção e rastreamento (YOLO + DeepSort)
│   ├── pipeline.py         # Coordenação frame a frame do pipeline experimental
│   └── voting.py           # Votação majoritária por track_id
├── models/
│   ├── factory.py          # Instanciação dinâmica de wrappers de modelos
│   ├── deepface_models.py  # Wrapper para modelos nativos do DeepFace
│   └── onnx_models.py      # Wrapper genérico para modelos ONNX (iResNet-100)
├── metrics/
│   └── evaluator.py        # Cálculo de Accuracy, ROC/AUC, TAR@FAR, Precision/Recall@K
└── utils/
    ├── preprocessing.py    # Aplicação de filtros (CLAHE + redimensionamento)
    └── helpers.py          # Obtenção da raiz do projeto e utilitários
main.py                     # Script de entrada para execução do benchmark
tests/                      # Suíte de testes unitários e de integração (TDD)
```

---

## Pré-requisitos e Instalação

Este projeto utiliza o gerenciador de pacotes **uv** para gerenciar o ambiente de desenvolvimento e as dependências de forma ultra-rápida.

1. **Instalar o `uv`:** Garanta que a ferramenta `uv` esteja instalada em sua máquina.
2. **Sincronizar as dependências:** Antes de executar qualquer código ou testes, sincronize as dependências e o ambiente virtual executando:
   ```bash
   uv sync
   ```

---

## Como Executar o Experimento

Com as dependências sincronizadas, para rodar o benchmark completo sequencialmente sobre todos os modelos e vídeos disponíveis, execute:

```bash
uv run python main.py
```

### Resultados Gerados

As métricas são salvas na pasta `data/results/`:
- **`data/results/[modelo].csv`:** Detalha a classificação final de cada `track_id` mapeado, informando o Ground Truth, a classificação votada e a indicação de acerto/erro.
- **`data/results/data.csv`:** Resumo comparativo agregando métricas de todos os modelos (Acurácia de rastros, ROC/AUC frame a frame, TAR@FAR 1%, Recall@1/3, Precision@1/3 e tempo de inferência em ms).

---

## Como Executar os Testes Automatizados

O repositório segue rigorosamente práticas de TDD. Para validar a suíte de testes e obter a cobertura do código, execute:

```bash
uv run pytest
```
