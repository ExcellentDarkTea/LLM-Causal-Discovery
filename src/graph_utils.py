import pyAgrum as gum
import pandas as pd
import pyAgrum.lib.notebook as gnb
import pandas as pd
import numpy as np

def mediator_removal(A): #A = adj_matrix.copy().to_numpy()
    """
    Remove direct edges that can be mediated by other nodes.
    """
    n = len(A)
    A_new = [row[:] for row in A]

    for i in range(n):
        for j in range(n):
            # Check if there is a direct edge from i to j
            if i != j and A[i][j] == 1:
                # Check if there exists a mediator k such that i -> k and k -> j
                has_indirect_path = any(
                    A[i][k] == 1 and A[k][j] == 1
                    for k in range(n) if k != i and k != j
                )
                # If an indirect path exists through some k, remove the direct edge
                if has_indirect_path:
                    A_new[i][j] = 0

    return np.array(A_new)

def remove_true_bidirectional_conflicts(df: pd.DataFrame):
    """
    Remove bidirectional relationships from the DataFrame where one relationship based on probability   
    """

    df_filtered = df.copy()
    
    # Filter only TRUE relationships
    true_df = df[df['answer'] == True]

    to_remove = []

    for i, row in true_df.iterrows():
        var1, var2 = row['var1'], row['var2']
        prob1 = row['probability']

        reverse_match = true_df[
            (true_df['var1'] == var2) & 
            (true_df['var2'] == var1)
        ]
        
        if not reverse_match.empty:
            prob2 = reverse_match.iloc[0]['probability']

            if prob1 < prob2:
                to_remove.append(i)
            else:
                to_remove.append(reverse_match.index[0])

    # Remove marked rows
    df_filtered = df_filtered.drop(index=to_remove).reset_index(drop=True)
    return df_filtered

def create_adjacency_matrix(df, node_order):
    """
    Creates an adjacency matrix from a DataFrame of relationships.
    """
    adj_matrix = pd.DataFrame(0, index=node_order, columns=node_order)
    
    for _, row in df.iterrows():
        if row['answer']:  # only add TRUE relationships
            source = row['var1']
            target = row['var2']
            if source in node_order and target in node_order:
                adj_matrix.loc[source, target] = 1

    return adj_matrix
 

def create_bayesnet_from_adjacency(df):
    """
    Creates a Bayesian network in pyAgrum from a pandas DataFrame adjacency matrix.

    Parameters:
    - df (pd.DataFrame): A square DataFrame where rows and columns are node names,
                         and values are 0 (no edge) or 1 (edge).

    Returns:
    - bn (gum.BayesNet): The created Bayesian network with structure defined.
    """
    # Create an empty Bayesian network
    bn = gum.BayesNet("MedicalDiagnosisNetwork")
    
    # Add nodes (variables) to the network
    node_ids = {}
    for node in df.index:
        # Assume binary states (true/false) for all nodes
        var = gum.LabelizedVariable(node, node, 2)
        node_id = bn.add(var)
        node_ids[node] = node_id
    
    # Add arcs based on the adjacency matrix
    for i, row in df.iterrows():
        for j, value in row.items():
            if value == 1:
                parent_id = node_ids[i]
                child_id = node_ids[j]
                # print(f"Adding arc from {i} to {j}")
                try:
                    bn.addArc(parent_id, child_id)
                except gum.InvalidDirectedCycle:
                    print(f"Cycle detected: Cannot add arc from {i} to {j}. Skipping.")
    
    return bn