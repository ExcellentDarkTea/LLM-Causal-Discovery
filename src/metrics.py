import pandas as pd
import numpy as np
from collections import Counter
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from sklearn.metrics import f1_score
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon


def normalize_metrics(shd, tene, tere, original_matrix):
    """
    Normalize the SHD, TENE, and TERE scores based on matrix dimensions and edge counts.
    """

    total_elements = original_matrix.size
    true_edges_original = (original_matrix == 1).sum().sum()
    normalized_shd = shd / total_elements
    normalized_tene = tene / true_edges_original if true_edges_original > 0 else 0
    normalized_tere = tere / true_edges_original if true_edges_original > 0 else 0
    
    return normalized_shd, normalized_tene, normalized_tere


def evaluate_adjacency_matrices(original, new):
    """
    Evaluate similarity between two adjacency matrices using several metrics.
    """
    assert original.shape == new.shape, "Matrices must be of the same shape"

    # Flatten matrices
    original_flat = original.flatten()
    new_flat = new.flatten()
    
    # Structural Hamming Distance (SHD)
    shd = np.sum(original_flat != new_flat)

    # True Positives, False Positives, etc.
    tp = np.sum((original_flat == 1) & (new_flat == 1))
    fn = np.sum((original_flat == 1) & (new_flat == 0))
    fp = np.sum((original_flat == 0) & (new_flat == 1))
    tn = np.sum((original_flat == 0) & (new_flat == 0))

    # TENE: True Edge to Negative Edge
    tene = fn

    f1 = f1_score(original_flat, new_flat)
    # 2. True Edge to Negative Edge (TENE)
    true_edges_original = original == 1
    missing_edges = (true_edges_original & (new == 0)).sum()
    
    # 3. True Edge to Reverse Edge (TERE)

    true_edges = np.argwhere(original == 1)
    reverse_edges = 0

    for (i, j) in true_edges:
        if new[j, i] == 1:  # Check reverse edge
            reverse_edges += 1
            print(f"!!!!!!!!!!!!!Reverse edge found between {i} and {j}")

    normalized_shd, normalized_tene, normalized_tere = normalize_metrics(shd, missing_edges, reverse_edges, original)

    return {
        "SHD":          round(shd,4),
        "SHD (normalized)": round(normalized_shd,4),
        "TENE":         round(missing_edges,4),
        "TENE (normalized)": round(normalized_tene,4),
        "TERE":         round(reverse_edges,4),
        "TERE (normalized)": round(normalized_tere,4),
        "F1 Score":     round(f1,4), 
    }


def extract_short_version(s):
    start = s.find("'") + 1
    end = s.find("'", start)
    return s[start:end]  

def clean_dataframe(df):
    df["answer_binary"] = df["answer"].astype(bool).astype(int)
    df = df[~df["verb"].str.contains("NOT", na=False)].reset_index(drop=True)
    df["var1"] = df["var1"].apply(extract_short_version).str.replace(" ", "_")
    df["var2"] = df["var2"].apply(extract_short_version).str.replace(" ", "_")
    df["verb"] = df["verb"].str.replace(" ", "_")
    return df

def evaluate_and_store(adj_matrix, label_type, verb, var_unique, origin_matrix, store_dict, index_list, results_df):
    result = evaluate_adjacency_matrices(origin_matrix.to_numpy(), adj_matrix.to_numpy())
    result["verb"] = verb
    result["type"] = label_type
    index = f"{label_type}_{verb}".replace(" ", "_")
    
    index_list.append(index)
    store_dict[index] = adj_matrix.copy()
    
    result_row = pd.DataFrame([result], columns=results_df.columns)
    return pd.concat([results_df, result_row], ignore_index=True)

def majority_vote(answers):
    # Group by (var1, var2, verb) and get the most common answer
    count = Counter(answers)
    return count.most_common(1)[0][0]



def significant_check (df_combined, column_name, p_value, data, verb_list):
    
    df_results = pd.DataFrame(columns=["verb1", "verb2", "statistic", "p_value", "significant", "data"])
            
    if p_value < 0.05:
        print("Friedman test is significant, performing post-hoc tests")
        # Perform Wilcoxon signed-rank tests for all pairs of verbs
        for i in range(len(verb_list)):
            for j in range(i + 1, len(verb_list)):
                verb1 = verb_list[i]
                verb2 = verb_list[j]
                stat, p = wilcoxon(
                    df_combined[df_combined["verb"] == verb1][column_name],
                    df_combined[df_combined["verb"] == verb2][column_name]
                )

                df_results = pd.concat([df_results, pd.DataFrame({
                    "verb1": [verb1],
                    "verb2": [verb2],
                    "statistic": [stat],
                    "p_value": [p],
                    "significant": [p < 0.05],
                    "data": [data]
                })], ignore_index=True)

 
                # print(f"Wilcoxon test between {verb1} and {verb2}: statistic={stat}, p-value={p}")
                if p < 0.05:
                    print(f"Significant difference between {verb1} and {verb2}")
    else:
        print("The distributions are not significantly different., no post-hoc tests needed")
    return df_results  