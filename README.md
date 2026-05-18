# Will AI Take the Hint? Probing Referential Availability in Implied Discourse

This repository contains the code for the project and dataset creation for the Bachelor's thesis: **"Will AI Take the Hint? Probing Referential Availability in Implied Discourse Using the Hidden States of Large Language Models"** (Tilburg University). 

This project investigates whether Large Language Models (LLMs) encode the referential availability of discourse entities (and specifically when availability states are implied), and whether they use this internal representation for their behavioural output.

## Required Packages

All behavioural and probing analyses were implemented using Python 3.12.3. The following core packages were utilized:
* **PyTorch 2.6.0** (Paszke et al., 2019) with ROCm 6.1 for model inference
* **HuggingFace Transformers 5.2.0** (Wolf et al., 2020) for loading the models
* **NumPy 2.3.5** (Harris et al., 2020) for array manipulation and storing
* **pandas 3.0.1** (pandas development team, 2020) for data handling
* **scikit-learn 1.8.0** (Pedregosa et al., 2011) for probing classifiers

## Dataset (RAID)

The dataset can be generated locally using the `generate_raid.py` file. Alternatively, all information about the dataset and the direct download can be found on Hugging Face: **[tbogaers/RAID on Hugging Face](https://huggingface.co/datasets/tbogaers/RAID)**

## Code Structure & Usage

### 1. Optimal L2 Regularization (Grid Search)
Optimal L2 Regularization (C) for the linear probes was found using Cross-Validation. 
* Run `gridsearch.py` to calculate the optimal C-value for each probe (note that each layer requires its own uniquely trained probe).

### 2. GPT-2 XL Pipeline
* Run `gpt2xl.py`. This script is an end-to-end pipeline that includes the extraction of the hidden states, training and testing of the linear probes, and the behavioral evaluation for the GPT-2 XL model.

### 3. Llama 3.1 8B Pipeline
* Run `llama318b.py`. Similar to the GPT-2 script, this includes extraction of the hidden states, training and testing of the probes, and behavioral evaluation for Llama 3.1 8B. *Note that Llama 3.1 8B hidden state extraction takes a lot of vram, their model recommends 16GB of vram available.*
