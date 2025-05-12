# LLM-Causal-Discovery
# 🔍 LLM-based Causal Discovery Toolkit

This project explores the use of Large Language Models (LLMs) for causal discovery on well-established medical benchmark datasets. By leveraging prompt-based techniques, the notebook investigates whether LLMs can infer causal relationships between variables in classic Bayesian networks like Cancer, Asia and Medical Diagnostics.

---
## 📌 Project Goals

This work is part of an exploration into **LLMs as tools for expert elicitation**—particularly for building causal models in domains like healthcare, where manual modeling is time-consuming and costly.

## 💡 Project Summary

**Business/Research Context:**  
Understanding causal relationships is critical in domains like healthcare for decision support systems. Traditional statistical methods often fall short in capturing nuanced, real-world causal links. This project combines recent advances in Large Language Models (LLMs), such as GPT, with causal graph modeling to perform discovery, validation, and sensitivity analysis of causal structures from data.

This project aims to address two central questions:

1. How can LLMs contribute to causal discovery without becoming a bottleneck for reliability? That is, how can we evaluate and trust their outputs beyond simple accuracy metrics or downstream task performance?

2. Which prompt designs and linguistic choices improve the LLM’s ability to detect or validate causal relationships? In particular, how does verb framing, question format, or entity context affect the model's confidence and consistency?

---


## 📂 Project Structure

* **`LLM-for-Causal-Discovery.ipynb`**: Main notebook where LLMs are queried using crafted prompts to discover causal relations among variables in classic Bayesian networks.
* **`Validate-adj-matrix.ipynb`**: Evaluation module that compares the inferred causal adjacency matrix with the ground truth, using several structural metrics.

```
LLM-Causal-Discovery/
├── data/                             # Benchmark datasets used for causal discovery
│   ├── cancer.csv
│   ├── asia.csv
│   └── medical-diagnosis.csv
├── LLM-for-Causal-Discovery.ipynb       # LLM-based causal graph construction
├── Validate-adj-matrix.ipynb            # Graph evaluation metrics
├── Verb-Sensitivity-Analysis.ipynb      # Prompt robustness analysis  
├── src/                              # Source code for LLM prompting and graph utilities
│   ├── llm_prompting.py                        # Functions to query LLM with causal questions
│   ├── graph_utils.py                          # Adjacency matrix and visualization helpers
│   ├── metrics.py                              # Evaluation metric calculations
│   └── verb_variation.py                       # Verb generation and standardization
├── results/                          # Output results and visualizations
│   ├── inferred_graphs/
│   ├── evaluation_scores/
│   └── sensitivity_results/
├── models/                           # (Optional) Saved LLM outputs or vectorized results
│   └── cached_llm_responses.pkl
├── README.md                         # Project overview and instructions
├── requirements.txt                  # Python dependencies

```

---

## 📊 Datasets Used

* **Cancer**: Includes variables like Pollution, Smoker, Cancer, Dyspnoea, and X-ray.
* **Asia**: A respiratory disease-related network including Asia travel, Tuberculosis, Smoking, Lung Cancer, Bronchitis, X-ray, and Dyspnoea.
* **Medicine Diagnistic**: A cardiovascular disease-related network including: Eat Fatty Food, Arteriosclerosis,	Right Heart Syndrome,	Left Heart Syndrome,	Lungs Sound Funny,	Difficulty Breathing,	Smoking,	Radon Exposure,	Lung Cancer,	Cough Up Blood

These datasets are standard in causal inference benchmarking and have clearly defined ground-truth structures.

---

## 🤖 Methodology

### Causal Inference with LLMs

* Uses **prompt engineering** to ask LLMs about possible causal links (e.g., "Does smoking cause cancer?").
* Applies various **causal verb formulations** such as “lead to”, “influence”, “is caused by” to evaluate robustness.
* Aggregates LLM responses into a **binary adjacency matrix**, representing the inferred causal structure.

### Graph Evaluation

The predicted matrices are evaluated against the ground truth using:

* **Structural Hamming Distance (SHD)**: Counts mismatched edges.
* **TENE** (True Edge to Negative Edge): Missed true edges.
* **TERE** (True Edge to Reverse Edge): Reversed directions in predicted edges.
* **F1 Score**: Measures the balance between precision and recall.
* **Normalized Metrics**: Adjusted versions of the above based on matrix size and edge count.

These metrics are computed by the utility functions in `Validate-adj-matrix.ipynb`.



---

## 📦 Requirements

Install the required libraries with:

```bash
pip install -r requirements.txt
```

Main dependencies:

* `openai` (or another LLM provider)
* `pandas`, `numpy`
* `scikit-learn`
* `networkx`, `matplotlib`

---



