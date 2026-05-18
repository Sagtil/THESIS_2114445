# Llama 3.1 8B Pipeline, Ties Bogaers (2114445)

import torch
import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
from transformers import AutoTokenizer, AutoModelForCausalLM

# Seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration, batch size for 4 prompts at a time during hidden state extraction
DATASET_PATH = "DATASET/RAID.csv"
OUTPUT_DIR = "RESULTS/HIDDENSTATES"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_NAME = "meta-llama/Llama-3.1-8B"
MODEL_KEY = "llama-3.1-8b"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split into probe training, held-out poles, and generalisation
# Train: 80% of explicit_leave + baseline
# Test: 20% of explicit_leave + baseline lexical held-out test set
# Generalisation: implied_leave, implied_cancel and disengaged 
df = pd.read_csv(DATASET_PATH)
probe_train = df[df["split"] == "train"].reset_index(drop=True)
probe_eval_poles = df[df["split"] == "test"].reset_index(drop=True)
probe_eval_gen = df[df["split"] == "generalisation"].reset_index(drop=True)

# All evaluation data combined (for hidden state extraction)
probe_eval = df[df["split"].isin(["test", "generalisation"])].reset_index(drop=True)

# Map probe training conditions to binary labels
# 1 = one antecedent available
# 0 = two antecedents available (ambiguous)
LABEL_MAP = {"explicit_leave": 1, "baseline": 0}
probe_train["label_id"] = probe_train["condition"].map(LABEL_MAP)

# Extract character names using RegEx, name1 and name2 (the 2 male characters). Important for behavioiural evaluation.
# Prompts have 3 different patterns to try, because of different sentence strucutres
def extract_names(prompt):
    match = re.search(r"(?:talking|chatting|conversation) (?:to|with) (\w+) and (\w+)", prompt)
    if match:
        return match.group(1), match.group(2)
    match = re.search(r"(\w+), (\w+), and (\w+) were talking", prompt)
    if match:
        return match.group(2), match.group(3)
    match = re.search(r"with (\w+) and (\w+),? (?:deep in|having)", prompt)
    if match:
        return match.group(1), match.group(2)
    print(f"WARNING: Couldn't extract names from: {prompt[:80]}")
    return None, None

# Assign stayer to antecedent_label (the person who stays in conversation)
# If ambiguous, name1 automatically becomes stayer (for ambiguous sentences it does not matter)
for sub_df in [probe_train, probe_eval, df]:
    names = sub_df["prompt"].apply(extract_names)
    sub_df["name1"] = names.apply(lambda x: x[0])
    sub_df["name2"] = names.apply(lambda x: x[1])
    sub_df["stayer"] = np.where(
        sub_df["antecedent_label"] != "AMBIGUOUS",
        sub_df["antecedent_label"],
        sub_df["name1"],
    )
    # Assigns changer to the other character
    sub_df["changer"] = np.where(
        sub_df["antecedent_label"] != "AMBIGUOUS",
        np.where(sub_df["antecedent_label"] == sub_df["name1"], sub_df["name2"], sub_df["name1"]),
        sub_df["name2"],
    )

print("DATASET OVERVIEW")
print(f"\n{'Total samples:'} {len(df)}")
print(f"{'Probe training:'} {len(probe_train)} (explicit_leave + baseline)")
print(f"{'Probe evaluation (poles):'} {len(probe_eval_poles)} (held-out explicit_leave + baseline)")
print(f"{'Probe evaluation (generalisation):'} {len(probe_eval_gen)} (implied_leave + implied_cancel + disengaged)")
print(f"{'Probe evaluation (total):'} {len(probe_eval)}")

print(f"\nTraining label balance:")
print(probe_train["label_id"].value_counts().rename({1: "1 (explicit_leave)", 0: "0 (baseline)"}).to_string())

print(f"\nCondition * split:")
print(pd.crosstab(df["split"], df["condition"]).to_string())

print(f"\nName extraction check:")
for name, sub_df in [("train", probe_train), ("evaluation", probe_eval)]:
    nulls = sub_df["name1"].isna().sum()
    print(f"{name}: {nulls} failures out of {len(sub_df)}")

# Loading LLAMA 3.1 8B
# Filling empty space with eos tokens
print(f"Loading {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model loaded in native float16 precision
# If usage goes over 12GB vram, offloades to RAM (20GB)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "12GiB", "cpu": "20GiB"},
)
model.eval()


print(f"{'Layers:'} {model.config.num_hidden_layers}")
print(f"{'Hidden dim:'} {model.config.hidden_size}")
print(f"{'Dtype:'} {next(model.parameters()).dtype}")
print(f"{'Device map:'} {model.hf_device_map}")

# Locating the target pronoun "him"
def find_target_pronoun_position(prompt, tokenizer):
    # Find the token index of the first "him"
    # Avoiding it picks "him" from the question "Question: Who is 'him'? Answer:"
    # Tokenizer.encode converts prompt into list of token IDs
    # Tokenizer.decode convers token IDs into string representation (in case of words becoming multiple tokens)
    tokens = tokenizer.encode(prompt)
    token_strings = [tokenizer.decode([t]) for t in tokens]

    # Stripping white space and finding first "him"
    for i, tok_str in enumerate(token_strings):
        if tok_str.strip().lower() == "him":
            return i

    print(f"WARNING: Could not find 'him' in: {prompt[:80]}")
    return None

# Apply find target pronoun function to every row to get all token positions 
# Token position is the ideal position, moment of referential resolution 
for name, sub_df in [("train", probe_train), ("evaluation", probe_eval)]:
    positions = []
    failed = 0
    for _, row in sub_df.iterrows():
        pos = find_target_pronoun_position(row["prompt"], tokenizer)
        if pos is None:
            failed += 1
        positions.append(pos)
    sub_df["pronoun_pos"] = positions
    print(f"{name}: found {len(sub_df) - failed}/{len(sub_df)} pronouns ({failed} failures)")

# Sanity checks to see if 'him' gets found in all prompts (thus also in all conditions)
for cond in ["explicit_leave", "baseline", "implied_leave", "implied_cancel", "disengaged"]:
    row = df[df["condition"] == cond].iloc[0]
    pos = find_target_pronoun_position(row["prompt"], tokenizer)
    tokens = tokenizer.encode(row["prompt"])
    token_strings = [tokenizer.decode([t]) for t in tokens]

    # Show a window around the pronoun
    start = max(0, pos - 3)
    end = min(len(token_strings), pos + 4)
    window = token_strings[start:end]
    window_display = [f"\033[1m{t}\033[0m" if i + start == pos else t for i, t in enumerate(window)]

    print(f"{cond}:")
    print(f"Position {pos}: ...{''.join(window_display)}...")
    print()

def extract_hidden_states(model, tokenizer, df, batch_size=4):
    # 32 transformer layers and adding 1 for the embedding layer
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size
    num_samples = len(df)
    MAX_LENGTH = tokenizer.model_max_length

    # Sanity check so none of my prompts are too long
    lengths = df["prompt"].apply(lambda p: len(tokenizer.encode(p)))
    print(f"Prompt lengths: min={lengths.min()}, mean={lengths.mean():.0f}, max={lengths.max()}")
    print(f"Model max length: {MAX_LENGTH}")
    if lengths.max() > MAX_LENGTH:
        print(f"  WARNING: {(lengths > MAX_LENGTH).sum()} prompts exceed max length!")

    # 3D array to store shapes with number of samples x 33 layers x 4096 dimensions
    all_hidden_states = np.zeros((num_samples, num_layers, hidden_dim), dtype=np.float32)

    # Iterates through dataset in steps of 4 (batch size)
    # tdqm for progress bar
    for start in tqdm(range(0, num_samples, batch_size), desc="Extracting hidden states"):
        end = min(start + batch_size, num_samples)
        batch_df = df.iloc[start:end]
        prompts = batch_df["prompt"].tolist()
        # Tokenises the batch and converting to PyTorch tensors
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Hidden states, tuple of 33 tensors
        hidden_states = outputs.hidden_states

        for i, (_, row) in enumerate(batch_df.iterrows()):
            pos = row["pronoun_pos"]
            if pos is None:
                continue

            # Counts padding tokens and shifts pronoun position right to find correct position in the padded sequence
            if tokenizer.padding_side == "left":
                num_pad = (inputs["input_ids"][i] == tokenizer.pad_token_id).sum().item()
                adjusted_pos = pos + num_pad
            else:
                adjusted_pos = pos

            seq_len = inputs["input_ids"].shape[1]
            if adjusted_pos >= seq_len:
                print(f"  WARNING: position {adjusted_pos} >= seq_len {seq_len}, skipping")
                continue

            # Gives 4096-dimensional vector at the pronoun for the layer on
            # Changes to float32 when moving to CPU for numerical stability
            for layer_idx in range(num_layers):
                vec = hidden_states[layer_idx][i, adjusted_pos, :]
                all_hidden_states[start + i, layer_idx, :] = vec.cpu().float().numpy()
    # Complete (samples x 33 x 4096) array containing every hidden state vector, for every sample, for every layer
    return all_hidden_states


# Extract for training set
print("Extracting hidden states (train)")
hs_train = extract_hidden_states(model, tokenizer, probe_train, BATCH_SIZE)
print(f"Shape: {hs_train.shape}")

# Extract for evaluation set
print("\nExtracting hidden states (eval)")
hs_eval = extract_hidden_states(model, tokenizer, probe_eval, BATCH_SIZE)
print(f"Shape: {hs_eval.shape}")

# Save to disk
np.save(os.path.join(OUTPUT_DIR, f"hs_train_{MODEL_KEY}.npy"), hs_train)
np.save(os.path.join(OUTPUT_DIR, f"hs_eval_{MODEL_KEY}.npy"), hs_eval)
print("\nSaved to disk.")

# Best L2 regularization (C) values per layer found via 5-fold CV Grid Search
LLAMA_BEST_C = [
    0.001, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 
    0.01, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001, 0.01, 0.01, 0.01, 
    0.01, 0.01, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 
    0.1, 0.1, 0.1
]

# binary labels (0/1) and bootstrapping (1000 interations)
def train_and_evaluate_probes_bootstrap(hs_train, labels_train, hs_eval, conditions_eval, best_c_list, n_boot=1000):
    num_layers = hs_train.shape[1]
    results = []
    rng = np.random.default_rng(42) # Fixed seed for reproducibility

    for layer_idx in tqdm(range(num_layers), desc="Probing layers"):
        # Slice hidden states to current layer
        X_train = hs_train[:, layer_idx, :]
        X_eval = hs_eval[:, layer_idx, :]

        # Standardise features to zero mean and unit variance
        # Scaler fitted on training data only to prevent data leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

        # Retrieve the mathematically best 'C' value for this specific layer
        layer_c = best_c_list[layer_idx]

        # Train final probe on full training set with dynamic L2 regularisation
        clf = LogisticRegression(
            class_weight=None, solver="lbfgs",
            max_iter=1000, random_state=42, C=layer_c,
        )
        clf.fit(X_train, labels_train)
        y_pred = clf.predict(X_eval)

        for cond in np.unique(conditions_eval):
            mask = conditions_eval == cond
            preds = y_pred[mask]

            # For pole conditions: measure accuracy against ground truth
            # For generalisation conditions: measure alignment with theoretical expectations 
            if cond == "explicit_leave":
                scores = (preds == 1).astype(float)
                metric = "accuracy"
            elif cond == "baseline":
                scores = (preds == 0).astype(float)
                metric = "accuracy"
            elif cond in ("implied_leave", "disengaged"):
                scores = (preds == 1).astype(float)
                metric = "pct_predicted_1_available"
            elif cond == "implied_cancel":
                scores = (preds == 0).astype(float)
                metric = "pct_predicted_2_available"

            value = scores.mean()

            # Non-parametric bootstrap CI with 1000 resamples
            boot_means = []
            for _ in range(n_boot):
                idx = rng.choice(len(scores), size=len(scores), replace=True)
                boot_means.append(scores[idx].mean())
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)

            results.append({
                "layer": layer_idx,
                "condition": cond,
                "metric": metric,
                "value": value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "best_C_used": layer_c,
                "n_samples": int(mask.sum()),
            })

    return pd.DataFrame(results)

# Extract labels and conditions for probing
labels_train = probe_train["label_id"].values
conditions_eval = probe_eval["condition"].values

# Run full probing
probe_results_ci = train_and_evaluate_probes_bootstrap(
    hs_train, labels_train, hs_eval, conditions_eval, LLAMA_BEST_C
)
probe_results_ci.to_csv(os.path.join(OUTPUT_DIR, f"probe_results_ci_{MODEL_KEY}.csv"), index=False)

# Behavioural evaluation with model, tokenizer and dataframe as input
def evaluate_behavioral(model, tokenizer, df):
    # Compare probability the model assigns to stayer vs changer name after 'Answer:'.
    results = []

    # Iterates through each row one by one and tokenizes single prompt at a time
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating behaviour"):
        prompt = row["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Runs full forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits after "Answer:" in the prompt
        # Raw logits are converted to probablities across the full vocabulary
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=0)

        # Encoding stayer and changer names with leading space (tokens mid sentence have space prefix with Llama)
        stayer_tokens = tokenizer.encode(" " + row["stayer"], add_special_tokens=False)
        changer_tokens = tokenizer.encode(" " + row["changer"], add_special_tokens=False)
        # Probability of first token of each name
        stayer_prob = probs[stayer_tokens[0]].item()
        changer_prob = probs[changer_tokens[0]].item()

        # Ambiguous answers do not matter since both are correct
        if row["antecedent_label"] == "AMBIGUOUS":
            correct = None
            predicted_role = "stayer" if stayer_prob >= changer_prob else "changer"
        else:
            # Model is correct if stayer gets higher softmax probability
            true_name = row["antecedent_label"]
            if true_name == row["stayer"]:
                correct = stayer_prob > changer_prob
            else:
                correct = changer_prob > stayer_prob
            predicted_role = "stayer" if stayer_prob > changer_prob else "changer"

        results.append({
            "condition": row["condition"],
            "true_label": row["antecedent_label"],
            "stayer_prob": stayer_prob,
            "changer_prob": changer_prob,
            "predicted_role": predicted_role,
            "correct": correct,
        })

    return pd.DataFrame(results)

# Run on evaluation set
print("Evaluating behavioural probabilities")
behavioral_df = evaluate_behavioral(model, tokenizer, probe_eval)
behavioral_df.to_csv(os.path.join(OUTPUT_DIR, f"behavioral_{MODEL_KEY}.csv"), index=False)

# Non-ambiguous conditions accuracy
print("\nBehavioural accuracy per condition:")
for cond in ["explicit_leave", "implied_leave", "disengaged"]:
    cond_df = behavioral_df[behavioral_df["condition"] == cond]
    if len(cond_df) > 0:
        acc = cond_df["correct"].mean()
        print(f"  {cond:20s}: {acc:.3f} ({int(cond_df['correct'].sum())}/{len(cond_df)})")

# For ambiguous conditions which character does the model prefer
print("\nAmbiguous conditions (stayer preference):")
for cond in ["baseline", "implied_cancel"]:
    cond_df = behavioral_df[behavioral_df["condition"] == cond]
    if len(cond_df) > 0:
        stayer_pct = (cond_df["predicted_role"] == "stayer").mean()
        mean_diff = (cond_df["stayer_prob"] - cond_df["changer_prob"]).mean()
        print(f"  {cond:20s}: {stayer_pct:.3f} pick stayer, mean prob diff: {mean_diff:+.4f}")

# Behavioural results for R analysis
behavioral_export = behavioral_df.copy()
behavioral_export["role_order"] = probe_eval["role_order"].values
behavioral_export["structure"] = probe_eval["structure"].values
behavioral_export["s3_variant"] = probe_eval["s3_variant"].values
behavioral_export.to_csv(os.path.join(OUTPUT_DIR, f"behavioral_full_{MODEL_KEY}.csv"), index=False)

# Per-sample probe prediction  
def export_per_sample_predictions(hs_train, labels_train, hs_eval, probe_eval, model_key, best_c_list):
    num_layers = hs_train.shape[1]
    conditions_eval = probe_eval["condition"].values
    rows = []

    for layer_idx in tqdm(range(num_layers), desc="Exporting per-sample predictions"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(hs_train[:, layer_idx, :])
        X_eval = scaler.transform(hs_eval[:, layer_idx, :])

        layer_c = best_c_list[layer_idx]

        clf = LogisticRegression(
            class_weight=None, solver="lbfgs",
            max_iter=1000, random_state=42, C=layer_c,
        )
        clf.fit(X_train, labels_train)
        y_pred = clf.predict(X_eval)
        # Instead of binary like clf.predict, proba gives the actual probe probability per condition.
        y_prob = clf.predict_proba(X_eval)

        # It also saves sentence structure, role_order and there/not there in sentence 3 
        for i in range(len(probe_eval)):
            row = probe_eval.iloc[i]
            rows.append({
                "layer": layer_idx,
                "sample_idx": i,
                "condition": row["condition"],
                "structure": row["structure"],
                "role_order": row["role_order"],
                "s3_variant": row["s3_variant"],
                "prediction": int(y_pred[i]),
                "prob_2_available": float(y_prob[i, 0]),
                "prob_1_available": float(y_prob[i, 1]),
                "best_C_used": layer_c
            })

    out = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, f"per_sample_predictions_{model_key}.csv")
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows to {path}")

export_per_sample_predictions(hs_train, labels_train, hs_eval, probe_eval, MODEL_KEY, LLAMA_BEST_C)