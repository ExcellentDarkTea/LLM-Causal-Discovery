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

Compare predicted vs. true adjacency matrices using:

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
## Results:

The following tables summarize how different causal verb formulations influence the accuracy of the LLM-inferred adjacency matrices across three benchmark datasets: Cancer, Asia, and Mecidal.

**🔧 Structural Hamming Distance (SHD)**

| Verb                  |SHD (CANCER) | SHD (ASIA) | SHD (MECIDAL) | SHD (norm) (CANCER)  | SHD (norm) (ASIA) | SHD (norm) (MECIDAL)   |
|-----------------------|-------------|------------|---------------|----------------------|-------------------|------------------------|
| affect                |  0          | 11         | 8             | 0.00                 | 0.1719            | 0.08                   |
| increase the chance of|0            | 9          | 8             | 0.00                 | 0.1406            | 0.08                   |
| influence            | 0            | 5          | 10            | 0.00                 | 0.0781            | 0.10                   |
| lead to              | 0            | 7          | 8             | 0.00                 | 0.1094            | 0.08                   |
| raise the risk of    | 0            | 9          | 6             | 0.00                 | 0.1406            | 0.06                   |
| result in            | 0            | 4          | 10            | 0.00                 | 0.0625            | 0.10                   |
| **avg**              | 0            | 4          | 6             | 0.00                 | 0.0625            | 0.06                   |

**❌ True Edge to Negative Edge (TENE)**

| Verb                  | TENE (CANCER)  | TENE (ASIA) | TENE (MECIDAL) | TENE (norm) (CANCER)    | TENE (norm) (ASIA)  | TENE (norm) (MECIDAL)    |
|-----------------------|----------------|-------------|----------------|-------------------------|---------------------|--------------------------|
| affect               |  0              | 5           | 2              | 0.0                     | 0.625               | 0.2222                   |
| increase the chance of  0              | 4           | 3              | 0.0                     | 0.500               | 0.3333                   |
| influence            |  0              | 1           | 4              | 0.0                     | 0.125               | 0.4444                   |
| lead to              |  0              | 3           | 2              | 0.0                     | 0.375               | 0.2222                   |
| raise the risk of    |  0              | 4           | 1              | 0.0                     | 0.500               | 0.1111                   |
| result in            |  0              | 1           | 4              | 0.0                     | 0.125               | 0.4444                   |
| **avg**              |  0              | 1           | 1              | 0.0                     | 0.125               | 0.1111                   |
 

**✅ F1 Score**
 | Verb                   | F1 (CANCER) | F1 (ASIA) | F1 (MECIDAL) |
|------------------------|-------------|-----------|--------------|
| affect                 | 1.0000      | 0.3529    | 0.6364       |
| increase the chance of  | 1.0000      | 0.4706    | 0.6000       |
| influence              | 1.0000      | 0.7368    | 0.5000       |
| lead to                | 1.0000      | 0.5882    | 0.6364       |
| raise the risk of      | 1.0000      | 0.4706    | 0.7273       |
| result in              | 1.0000      | 0.7778    | 0.5000       |
| **avg**                | 1.0000      | 0.7778    | 0.7273       | 


* Robustness on Cancer dataset: All verbs result in perfect structural inference (SHD = 0, F1 = 1.0), suggesting that LLMs are highly reliable when querying simple networks with clear causal links.

* The Asia dataset showed the most variation across verb types. "Result in" and "influence" performed best (SHD: 4 and 5 respectively, F1: 0.7778 and 0.7368).

* Mecidal results were also verb-sensitive, with "raise the risk of" yielding the highest F1 (0.7273), while verbs like "influence" and "result in" led to lower performance.

* Verbs like “result in”, “influence”, and “lead to” consistently achieved lower SHD and higher F1 scores across datasets.

* High variability in TENE across verbs and datasets indicates that some formulations cause the LLM to miss true edges more frequently, particularly in Asia and Mecidal.


**A more detailed analysis of the results can be found in the article at the link ...**

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


