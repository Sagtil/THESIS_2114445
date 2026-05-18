# THESIS STATISTICS PIPELINE (FINAL GLM VERSION)
# Extracts exact numbers for the Results chapter using GLMs and Binomial Tests

library(tidyverse)
library(pROC)

# Set working directory (Change this to your actual folder)
setwd("C:/Users/tiesb/Documents/Thesis/tieses")

# --- Load Data ---
probe_llama <- read_csv("RESULTS/per_sample_predictions_llama-3.1-8b.csv", show_col_types = FALSE) %>% mutate(model = "Llama 3.1 8B")
probe_gpt2  <- read_csv("RESULTS/per_sample_predictions_gpt2-xl.csv", show_col_types = FALSE) %>% mutate(model = "GPT-2 XL")
probe_all <- bind_rows(probe_llama, probe_gpt2)

behav_llama <- read_csv("RESULTS/behavioral_full_llama-3.1-8b.csv", show_col_types = FALSE) %>% mutate(model = "Llama 3.1 8B")
behav_gpt2  <- read_csv("RESULTS/behavioral_full_gpt2-xl.csv", show_col_types = FALSE) %>% mutate(model = "GPT-2 XL")
behav_all <- bind_rows(behav_llama, behav_gpt2)



# 1. BEHAVIOURAL STATISTICS (Exact Binomial Test)

behav_stats <- behav_all %>% 
  filter(condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  group_by(model, condition) %>%
  summarise(
    total = n(),
    correct_count = sum(correct, na.rm = TRUE),
    accuracy = mean(correct, na.rm = TRUE),
    p_value = binom.test(correct_count, total, p = 0.5)$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    significance = case_when(p_value < 0.001 ~ "***", p_value < 0.01 ~ "**", p_value < 0.05 ~ "*", TRUE ~ "ns"),
    accuracy_pct = sprintf("%.1f%%", accuracy * 100)
  )

write_csv(behav_stats, "RESULTS/stats_behavioral_accuracy.csv")
print(behav_stats)



# 2. PROBING ACCURACY - ALL CONDITIONS (TRAINING POLES + GENERALISATION)

# Get peak layer (excluding layer 0) and final layer accuracy for ALL conditions
all_conditions_peak <- probe_all %>%
  filter(layer > 0) %>%
  mutate(correct = case_when(
    condition == "explicit_leave" ~ as.integer(prediction == 1),
    condition == "baseline"       ~ as.integer(prediction == 0),
    condition == "implied_leave"  ~ as.integer(prediction == 1),
    condition == "implied_cancel" ~ as.integer(prediction == 0),
    condition == "disengaged"     ~ as.integer(prediction == 1)
  )) %>%
  group_by(model, condition, layer) %>%
  summarise(accuracy = mean(correct), .groups = "drop") %>%
  group_by(model, condition) %>%
  summarise(
    peak_layer     = layer[which.max(accuracy)],
    peak_accuracy  = max(accuracy),
    final_layer    = max(layer),
    final_accuracy = accuracy[layer == max(layer)],
    .groups = "drop"
  )

# Binomial tests at final layer for ALL conditions
all_conditions_final <- probe_all %>%
  filter(layer > 0) %>%
  mutate(correct = case_when(
    condition == "explicit_leave" ~ as.integer(prediction == 1),
    condition == "baseline"       ~ as.integer(prediction == 0),
    condition == "implied_leave"  ~ as.integer(prediction == 1),
    condition == "implied_cancel" ~ as.integer(prediction == 0),
    condition == "disengaged"     ~ as.integer(prediction == 1)
  )) %>%
  group_by(model, condition) %>%
  filter(layer == max(layer)) %>%
  summarise(
    total         = n(),
    correct_count = sum(correct),
    accuracy      = mean(correct),
    p_value       = binom.test(correct_count, total, p = 0.5)$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    significance = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      TRUE            ~ "ns"
    ),
    accuracy_pct = sprintf("%.1f%%", accuracy * 100)
  )

print("Peak and final layer accuracy (all conditions):")
print(all_conditions_peak)
print("Final layer significance (all conditions):")
print(all_conditions_final)

write_csv(all_conditions_peak,  "RESULTS/stats_all_conditions_peak.csv")
write_csv(all_conditions_final, "RESULTS/stats_all_conditions_final.csv")



# 3. PROBING AUC (Area Under ROC Curve)


calc_auc <- function(df, target_cond, model_name) {
  sub_df <- df %>% filter(model == model_name, condition %in% c(target_cond, "baseline")) %>%
    mutate(label = ifelse(condition == target_cond, 1, 0))
  roc_obj <- roc(sub_df$label, sub_df$prob_1_available, quiet = TRUE)
  return(as.numeric(roc_obj$auc))
}

auc_results <- tibble(
  model = c("Llama 3.1 8B", "Llama 3.1 8B", "GPT-2 XL", "GPT-2 XL"),
  condition = c("explicit_leave", "implied_leave", "explicit_leave", "implied_leave"),
  AUC = c(
    calc_auc(probe_final, "explicit_leave", "Llama 3.1 8B"),
    calc_auc(probe_final, "implied_leave", "Llama 3.1 8B"),
    calc_auc(probe_final, "explicit_leave", "GPT-2 XL"),
    calc_auc(probe_final, "implied_leave", "GPT-2 XL")
  )
) %>% mutate(AUC_formatted = sprintf("%.3f", AUC))

write_csv(auc_results, "RESULTS/stats_probing_auc.csv")



# 4. STRUCTURAL CONFOUNDS (GLM - Logistic Regression)


confound_stats <- probe_final %>%
  filter(condition == "implied_leave") %>%
  mutate(correct = as.integer(prediction == 1)) %>%
  group_by(model) %>%
  summarise(
    structure_p_val = {
      sub_df <- cur_data()
      mdl <- glm(correct ~ structure + role_order, data = sub_df, family = binomial)
      drop1(mdl, test="Chisq")["structure", "Pr(>Chi)"]
    },
    role_order_p_val = {
      sub_df <- cur_data()
      mdl <- glm(correct ~ structure + role_order, data = sub_df, family = binomial)
      drop1(mdl, test="Chisq")["role_order", "Pr(>Chi)"]
    },
    .groups = "drop"
  ) %>%
  mutate(
    structure_significant = ifelse(structure_p_val < 0.05, "Yes (Biased)", "No (Fair)"),
    role_order_significant = ifelse(role_order_p_val < 0.05, "Yes (Biased)", "No (Fair)")
  )

write_csv(confound_stats, "RESULTS/stats_structural_confounds.csv")
print(confound_stats)



# 5. DIVERGENCE STATISTICS (Peak & Final Layer Exact Binomial Test)


# 1. Calculate Divergence and Total Correct for ALL layers to find the peak
layerwise_divergence <- probe_all %>%
  filter(condition %in% c("implied_leave", "implied_cancel")) %>%
  mutate(correct = case_when(
    condition == "implied_leave" ~ as.integer(prediction == 1),
    condition == "implied_cancel" ~ as.integer(prediction == 0)
  )) %>%
  group_by(model, layer) %>%
  summarise(
    total_trials = n(),
    total_correct = sum(correct),
    leave_acc = mean(correct[condition == "implied_leave"]),
    cancel_acc = mean(correct[condition == "implied_cancel"]),
    divergence = leave_acc + cancel_acc - 1,
    .groups = "drop"
  )

# 2. Extract Final Layer and Peak Layer for each model
divergence_stats <- layerwise_divergence %>%
  group_by(model) %>%
  mutate(
    is_final = (layer == max(layer)),
    is_peak = (divergence == max(divergence))
  ) %>%
  filter(is_final | is_peak) %>%
  mutate(
    evaluation_point = case_when(
      is_final & is_peak ~ "Peak & Final Layer",
      is_final ~ "Final Layer",
      is_peak ~ "Peak Layer"
    )
  ) %>%
  select(model, evaluation_point, layer, total_correct, total_trials, leave_acc, cancel_acc, divergence) %>%
  ungroup() %>%
  distinct(model, evaluation_point, .keep_all = TRUE) # Removes duplicates if there are ties

# 3. Calculate P-values using Exact Binomial Tests
# (Divergence > 0 is mathematically identical to Overall Accuracy > 50%)
divergence_stats <- divergence_stats %>%
  rowwise() %>%
  mutate(
    p_value = binom.test(total_correct, total_trials, p = 0.5)$p.value
  ) %>%
  ungroup() %>%
  mutate(
    significance = case_when(
      p_value < 0.001 ~ "***", 
      p_value < 0.01 ~ "**", 
      p_value < 0.05 ~ "*", 
      TRUE ~ "ns"
    )
  ) %>%
  select(-total_correct, -total_trials) %>% # Clean up helper columns
  arrange(model, desc(evaluation_point))

write_csv(divergence_stats, "RESULTS/stats_divergence.csv")
print(divergence_stats)


# 6. RECENCY BIAS STATISTICS (Exact Binomial Test)

recency_stats <- behav_all %>%
  filter(condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  mutate(
    second_mentioned = case_when(role_order == "changer_first" ~ "stayer", role_order == "stayer_first" ~ "changer"),
    picked_second = (predicted_role == second_mentioned)
  ) %>%
  group_by(model, condition) %>%
  summarise(
    total = n(),
    picked_second_count = sum(picked_second, na.rm=TRUE),
    pct_second = mean(picked_second, na.rm=TRUE),
    p_value = binom.test(picked_second_count, total, p = 0.5)$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    significance = case_when(p_value < 0.001 ~ "***", p_value < 0.01 ~ "**", p_value < 0.05 ~ "*", TRUE ~ "ns"),
    pct_formatted = sprintf("%.1f%%", pct_second * 100)
  )

write_csv(recency_stats, "RESULTS/stats_recency_bias.csv")


# 7. ROLE ORDER EFFECT ON PROBING

role_probe_stats <- probe_final %>%
  filter(condition == "implied_leave") %>%
  mutate(correct = as.integer(prediction == 1)) %>%
  group_by(model, role_order) %>%
  summarise(
    accuracy = mean(correct),
    n = n(),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = role_order, values_from = c(accuracy, n))

write_csv(role_probe_stats, "RESULTS/stats_role_order_probe.csv")


# 8. AUC: IMPLIED LEAVE vs IMPLIED CANCEL (Context Sensitivity)

calc_auc_leave_cancel <- function(df, model_name) {
  sub_df <- df %>%
    filter(model == model_name, condition %in% c("implied_leave", "implied_cancel")) %>%
    mutate(label = ifelse(condition == "implied_leave", 1, 0))
  roc_obj <- roc(sub_df$label, sub_df$prob_1_available, quiet = TRUE)
  ci_obj <- ci.auc(roc_obj, conf.level = 0.95)
  return(tibble(
    model = model_name,
    comparison = "implied_leave vs implied_cancel",
    AUC = as.numeric(roc_obj$auc),
    AUC_ci_low = as.numeric(ci_obj[1]),
    AUC_ci_high = as.numeric(ci_obj[3])
  ))
}

auc_context <- bind_rows(
  calc_auc_leave_cancel(probe_final, "Llama 3.1 8B"),
  calc_auc_leave_cancel(probe_final, "GPT-2 XL")
) %>% mutate(AUC_formatted = sprintf("%.3f [%.3f, %.3f]", AUC, AUC_ci_low, AUC_ci_high))

write_csv(auc_context, "RESULTS/stats_auc_context_sensitivity.csv")



# 9. PEAK LAYER IDENTIFICATION

peak_layers <- probe_all %>%
  filter(layer > 0) %>%  
  mutate(correct = case_when(
    condition == "explicit_leave" ~ as.integer(prediction == 1),
    condition == "baseline"       ~ as.integer(prediction == 0),
    condition == "implied_leave"  ~ as.integer(prediction == 1),
    condition == "implied_cancel" ~ as.integer(prediction == 0),
    condition == "disengaged"     ~ as.integer(prediction == 1)
  )) %>%
  group_by(model, condition, layer) %>%
  summarise(accuracy = mean(correct), .groups = "drop") %>%
  group_by(model, condition) %>%
  summarise(
    peak_layer = layer[which.max(accuracy)],
    peak_accuracy = max(accuracy),
    final_layer = max(layer),
    final_accuracy = accuracy[layer == max(layer)],
    .groups = "drop"
  )

write_csv(peak_layers, "RESULTS/stats_peak_layers.csv")
print(peak_layers)


# NAME-LEVEL OVERRIDE STATISTICS

name_override_stats <- behav_all %>%
  filter(condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  mutate(
    first_role = ifelse(role_order == "changer_first", "changer", "stayer"),
    picked_first = (predicted_role == first_role)
  ) %>%
  group_by(model, true_label) %>%
  summarise(
    n = n(),
    override_count = sum(picked_first, na.rm = TRUE),
    override_rate = mean(picked_first, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n >= 10) %>%
  group_by(model) %>%
  summarise(
    n_names = n(),
    mean_override = mean(override_rate),
    median_override = median(override_rate),
    sd_override = sd(override_rate),
    min_override = min(override_rate),
    max_override = max(override_rate),
    # Names significantly above 0.5 (override recency)
    n_names_above_chance = sum(override_rate > 0.5),
    pct_names_above_chance = mean(override_rate > 0.5),
    # One sample t-test: is mean override rate different from 0.5?
    t_stat = t.test(override_rate, mu = 0.5)$statistic,
    p_value = t.test(override_rate, mu = 0.5)$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    significance = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      TRUE            ~ "ns"
    )
  )

print(name_override_stats)
write_csv(name_override_stats, "RESULTS/stats_name_override.csv")

# Also save per-name rates for reference
name_override_pername <- behav_all %>%
  filter(condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  mutate(
    first_role = ifelse(role_order == "changer_first", "changer", "stayer"),
    picked_first = (predicted_role == first_role)
  ) %>%
  group_by(model, true_label) %>%
  summarise(
    n = n(),
    override_rate = mean(picked_first, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n >= 10) %>%
  arrange(model, desc(override_rate))

write_csv(name_override_pername, "RESULTS/stats_name_override_pername.csv")


