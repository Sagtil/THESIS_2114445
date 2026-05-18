# Grid Search Pipeline for Regularization, Ties Bogaers (2114445)

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Configuration (Matches your existing pipeline)
DATASET_PATH = "DATASET/RAID.csv"
OUTPUT_DIR = "RESULTS/HIDDENSTATES"
MODEL_KEY = "llama-3.1-8b" # or "gpt2-xl" 

# 1. Load the dataset just to get the training labels
print("Loading dataset labels...")
df = pd.read_csv(DATASET_PATH)
probe_train = df[df["split"] == "train"].reset_index(drop=True)

# 1 = one antecedent available, 0 = two antecedents available (ambiguous)
LABEL_MAP = {"explicit_leave": 1, "baseline": 0}
probe_train["label_id"] = probe_train["condition"].map(LABEL_MAP)
labels_train = probe_train["label_id"].values

# 2. Load the hidden states you already extracted and saved
# This skips the heavy LLM processing!
print(f"Loading hidden states for {MODEL_KEY} from disk...")
hs_train = np.load(os.path.join(OUTPUT_DIR, f"hs_train_{MODEL_KEY}.npy"))
print(f"Loaded train shape: {hs_train.shape}")

# 3. Grid Search Function
def grid_search_probes(hs_train, labels_train):
    num_layers = hs_train.shape[1]
    best_params_per_layer = []

    # We will test a range of C values (inverse of regularization strength)
    # Smaller C = stronger L2 regularization. 
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}

    for layer_idx in tqdm(range(num_layers), desc="Grid Searching Layers"):
        # Slice hidden states to current layer
        X_train = hs_train[:, layer_idx, :]

        # Standardise features to zero mean and unit variance
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Base probe model
        clf = LogisticRegression(
            class_weight="balanced", 
            solver="lbfgs", 
            max_iter=1000, # Increased slightly to ensure convergence
            random_state=42
        )

        # Grid search with 5-fold cross-validation
        # n_jobs=-1 uses all your CPU cores to make this run very fast
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, labels_train)

        best_params_per_layer.append({
            "layer": layer_idx,
            "best_C": grid_search.best_params_['C'],
            "best_cv_accuracy": grid_search.best_score_
        })

    return pd.DataFrame(best_params_per_layer)

# 4. Run and Save
print("Starting Grid Search...")
grid_search_results = grid_search_probes(hs_train, labels_train)
output_file = os.path.join(OUTPUT_DIR, f"grid_search_best_params_{MODEL_KEY}.csv")
grid_search_results.to_csv(output_file, index=False)

print(f"Grid search complete! Results saved to: {output_file}")
print("\nPreview of best parameters:")
print(grid_search_results.head())