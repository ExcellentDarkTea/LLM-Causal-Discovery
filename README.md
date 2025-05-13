# 🔍 LLM-based Causal Discovery Toolkit

This project investigates the potential of Large Language Models (LLMs) for causal discovery using prompt-based querying. Specifically, it explores whether LLMs can accurately infer causal graphs (Bayesian networks) over benchmark datasets when asked about causal relationships in natural language.

The script provided is in relation to the following paper: (arXive link for future)

---
# 💡 Project Summary

**Business/Research Context:**  
Understanding causal relationships is critical in domains like healthcare for decision support systems. Traditional statistical methods often fall short in capturing nuanced, real-world causal links. This project combines recent advances in LLMs, such as GPT, with causal graph modeling to perform discovery, validation, and sensitivity analysis of causal structures from data.

## 📌 Research Goals

* Evaluate whether LLMs can serve as **domain experts** for **expert elicitation** in causal discovery.
* Analyze **linguistic biases** in LLM behavior during causal reasoning tasks.
* Provide a reproducible framework for combining NLP and causal inference.
---


# 📂 Project Structure

* **`LLM-for-Causal-Discovery.ipynb`**: Core notebook where causal relationships are queried via LLM prompts and adjacency matrices are constructed.
* **`Validate-adj-matrix.ipynb`**: Contains utility functions for comparing predicted and ground-truth adjacency matrices using standard evaluation metrics.
* **`Verb-Sensitivity-Analysis.ipynb`**: Analyzes how different causal verbs influence LLM outputs, using statistical testing to assess sensitivity and robustness.

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
│   ├── graph_utils.py                    # Adjacency matrix and visualization helpers
│   └── metrics.py                        # Evaluation metric calculations
├── results/                          # Output results and visualizations
│   ├── asia/
│   ├── cancer/
│   └── medicine/
├── README.md                         # Project overview and instructions
├── requirements.txt                  # Python dependencies

```

---

## 📊 Datasets Used

* **Cancer**: Includes variables like Pollution, Smoker, Cancer, Dyspnoea, and X-ray.
* **Asia**: A respiratory disease-related network including Asia travel, Tuberculosis, Smoking, Lung Cancer, Bronchitis, X-ray, and Dyspnoea.
* **Medicine Diagnistic**: A cardiovascular disease-related network including: Eat Fatty Food, Arteriosclerosis,	Right Heart Syndrome,	Left Heart Syndrome,	Lungs Sound Funny,	Difficulty Breathing,	Smoking,	Radon Exposure,	Lung Cancer,	Cough Up Blood

These [datasets](https://www.bnlearn.com/bnrepository/) are standard in causal inference benchmarking and have clearly defined ground-truth structures.

---

## 🤖 Methodology

### Causal Inference with LLMs

* Uses **prompt engineering** to ask LLMs about possible causal links (e.g., "Does smoking cause cancer?").
* Applies various **causal verb formulations** such as “lead to”, “influence”, “is caused by” to evaluate robustness.
* Aggregates responses into an **adjacency matrix** representing the discovered causal graph.

### Graph Evaluation

Compares predicted vs. true adjacency matrices using:

* **Structural Hamming Distance (SHD)**: Counts mismatched edges.
* **TENE** (True Edge to Negative Edge): Missed true edges.
* **TERE** (True Edge to Reverse Edge): Reversed directions in predicted edges.
* **F1 Score**: Measures the balance between precision and recall.
* **Normalized Metrics**: Adjusted versions of the above based on matrix size and edge count.

These metrics are computed by the utility functions in `Validate-adj-matrix.ipynb`.

### Verb Sensitivity Analysis

* Evaluates the **impact of different causal verbs** on the output causal graphs.
* Uses non-parametric statistical tests:
  * **Friedman test** to assess global differences among verb groups.
  * **Wilcoxon test** for pairwise comparisons.
* Results highlight how sensitive LLMs are to phrasing, an important consideration for prompt design.


---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Key packages:

* `openai`, `pandas`, `numpy`, `scikit-learn`
* `networkx`, `matplotlib`, `seaborn`
* `scipy` (for statistical tests)

---

## 📈 Future Work

* Expand to large networks such as **Alarm** and **Child**.
* Compare LLM-generated graphs with those from traditional algorithms (e.g., PC, GES, NOTEARS).


