# SAM: LLM-Based Semantic-Augmented Adaptive Multi-Interest Model for Sequential Recommendation

This repository contains the official PyTorch implementation for the paper: **SAM: LLM-Based Semantic-Augmented Adaptive Multi-Interest Model for Sequential Recommendation**.

Our paper has been submitted to The 2026 ACM Web Conference. We will update the paper link and BibTeX citation upon acceptance.

**[Paper]**  | **[Code](https://github.com/tsinghua-fib-lab/SAM)** 

## Introduction

Sequential recommendation is crucial for capturing users' dynamic preferences. However, existing methods often struggle with two main challenges: effectively disentangling rich semantic interests from sparse user behaviors, and seamlessly integrating these semantic interests with traditional ID-based models.

To address these challenges, we propose **SAM**, a novel two-stage framework that leverages Large Language Models (LLMs) for semantic-augmented multi-interest learning. SAM excels at semantic understanding, applying them directly to multi-interest learning by generating a semantically-rich interest representation and employing a sophisticated fusion method to preserve the structural integrity of the ID embedding space.

<p align="center">
  <img src="assets/figures/sam_overview.png" width="800" alt="SAM overview">
  <br>
  <em>Figure 1: The overall framework of SAM.</em>
</p>

### Key Contributions:

*   **Adaptive Interest Extraction**: We introduce an Interest Group Identification Module that adaptively determines the optimal number of interests for each user, using this as a data-driven constraint to guide LLM-based semantic interest generation.
*   **Interest-Guided Representation Enhancement**: A dual-attention mechanism, including an Interest-Aware Attention and a Cross-Interest Attention mechanism, effectively fuses the generated semantic information with ID-based embeddings while preserving embedding space integrity and modeling complex interest inter-dependencies.
*   **State-of-the-Art Performance**: Extensive experiments on six benchmark datasets demonstrate that SAM significantly outperforms existing state-of-the-art baselines, especially for cold-start users and long-tail items.

## Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tsinghua-fib-lab/SAM.git
    cd SAM
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Our code is built with Python 3.11 and PyTorch 2.2.1. Install all required packages using:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

We use five sub-datasets from the **Amazon Review Data (2018)** and a large-scale **Alibaba** dataset.

### Amazon Datasets

1. **Download** the 5-core review data and metadata from [Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/):
   - All Beauty
   - Arts Crafts & Sewing
   - Industrial & Scientific
   - Musical Instruments
   - Office Products

2.  **Preprocess**: Place the downloaded files in the `Data/` directory and run the preprocessing script. This script will filter users/items with less than 5 interactions and generate a unified data format.

    For example, to process the 'Musical_Instruments' dataset:
    ```bash
    # Make sure reviews_Musical_Instruments_5.json and meta_Musical_Instruments.json are in Data/
    cd Data
    python DataProcessing.py --dataset Musical_Instruments
    ```
    This will generate `Data/Musical_Instruments.txt` which will be used in the training process.

#### Alibaba Dataset

The Alibaba dataset is proprietary and used with permission. For academic research purposes, please contact the authors.

## How to Run

The training process of SAM consists of two stages:

### Stage 1: Multi-Interest Extraction

In this stage, we use the Interest Group Identification Module to determine the optimal number of interests (K) for each user and then leverage **Qwen-Turbo** to generate semantic interest representations.

**Requirements**:
- Local BERT model: Download `[bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)` and place it under `LLMs/`
- DashScope API key for [Qwen-Turbo](https://dashscope.aliyun.com/) access

Setup and run (example for Musical Instruments):
```bash
# 1) Prepare API key (replace with your own key)
export DASHSCOPE_API_KEY='sk-xxxxxxxxxxxxxxxx'

# 2) Generate interests for All Beaury dataset
python stage1_generate_interests.py \
  --dataset Musical_Instruments \
  --api_key "$DASHSCOPE_API_KEY" \
```

### Stage 2: Interest-Guided Representation Enhancement

This is the main training stage. The model learns to fuse the semantic interests from Stage 1 with the user's behavior sequence.

Run (example for Musical Instruments):
```bash
python main.py \
  --dataset Musical_Instruments \
  --epochs 400 \
  --batch_size 1024 \
  --learning_rate 0.001 \
  --item_dim 128 \
```

## Performance

SAM achieves state-of-the-art performance across all six datasets. The table below summarizes the main results (MRR@20, NDCG@20).

| Dataset                | Metric  | STOSA | BSARec  | SINE   | EIMF   | **SAM (Ours)** | **Improvement** |
| ---------------------- | ------- | ------ | ------ | ---------- | ------ | ------ | -------------- | --------------- |
| **All Beauty**         | MRR@20   |0.2180 |  0.2026 |  0.2463 |  0.2535 | **0.2615**     | **+3.14%**      |
|                        | NDCG@20 |  0.2853 |  0.2764 | 0.2985 | 0.3085| **0.3266**     | **+5.85%**      |
| **Arts Crafts & Sewing** | MRR@20   |  0.2885 |  0.3221 | 0.3159 |  0.3238| **0.3251**     | **+0.40%**      |
|                        | NDCG@20 |  0.3495|  0.3911 |  0.3907 | 0.3858 | **0.3936**     | **+0.64%**      |
| **Industrial & Scientific** | MRR@20   |  0.2136 |  0.1933 |  0.2195 | 0.1833 | **0.2225**     | **+1.37%**      |
|                        | NDCG@20 |  0.2689 |  0.2587 |  0.2858 | 0.2512 | ** 0.2873**     | **+0.52%**      |
| **Musical Instruments**  | MRR@20   |  0.2823 |  0.2814 |  0.2794 | 0.2811 | ** 0.2831**     | **+0.28%**      |
|                        | NDCG@20 | 0.3767 | 0.3512 | 0.3314 | 0.3447 | **0.3575**     | **+1.79%**      |
| **Office Products**      | MRR@20   | 0.2887 | 0.2864 | 0.2842 |  0.2883 | **0.2921**     | **+1.18%**      |
|                        | NDCG@20 | 0.3597 | 0.3582 | 0.3562 | 0.3521 | **0.3609**     | **+0.33%**      |
| **Alibaba**              | MRR@20   | 0.5327 | 0.4695 | 0.4031 | 0.4720 | **0.5709**     | **+7.17%**      |
|                        | NDCG@20 | 0.5982 | 0.5209 | 0.4416 | 0.5251 | **0.6024**     | **+0.70%**      |

## Project Structure

```
SAM/
├── Data/
│   ├── DataProcessing.py          # Preprocessing script for Amazon datasets
│   └── {dataset}.txt              # Processed dataset files (user_id, item_id, title)
├── Interests/
│   ├── interests_{dataset}.pt     # Semantic interest embeddings from Stage 1
│   └── interests_{dataset}.csv    # Generated interest descriptions
├── LLMs/
│   └── bert-base-uncased/         # Local BERT model (download separately)         
├── main.py                        # Stage 2 training script
├── model.py                       # Stage 2 Model architecture
├── stage1_generate_interests.py   # Stage 1 interest extraction
├── stage2_attention.py            # Dual-attention mechanisms
├── trainer.py                     # Training and evaluation logic
├── utils.py                       # Utility functions
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Citation

If you find our work useful for your research, please consider citing our paper. The BibTeX entry will be provided upon publication.

```bibtex
@inproceedings{yang2026sam,
  title={SAM: LLM-Based Semantic-Augmented Adaptive Multi-Interest Model for Sequential Recommendation},
  author={Yang, Fei and Liu, Bin and Xu, Ziru and Zhu, Han and Gao, Chen and Li, Yongli},
  booktitle={Proceedings of the ACM web conference 2026},
  year={2026}
}
```

## Acknowledgements

- Our implementation for the dynamic routing mechanism is inspired by the official code of [MIND](https://github.com/Wang-Yu-Qing/MIND)

##  Contact

For questions or collaborations, please contact:

- **Fei Yang**: fei.yang@mail.bnu.edu.cn
- **Bin Liu**: zhuoli.lb@alibaba-inc.com
