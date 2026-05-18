  # THESIS RESULTS PIPELINE: REPRESENTATION & GENERATION
  # Models: Llama 3.1 8B & GPT-2 XL
  
  # %% Setup and Imports
  library(tidyverse)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(purrr)
  library(tidyr) 
  library(ggrepel)
  library(reticulate)
  library(lsa) 
  library(reshape2) 
  library(MASS)     
  library(patchwork)
  
  # Set working directory 
  setwd("C:/Users/tiesb/Documents/Thesis/tieses")
  
  # Import python's numpy directly into R for .npy files
  np <- import("numpy")
  
  # %% Load Data: CSVs
  ci_llama <- read_csv("RESULTS/probe_results_ci_llama-3.1-8b.csv", show_col_types = FALSE)
  ci_gpt2  <- read_csv("RESULTS/probe_results_ci_gpt2-xl.csv", show_col_types = FALSE)
  probe_llama <- read_csv("RESULTS/per_sample_predictions_llama-3.1-8b.csv", show_col_types = FALSE)
  probe_gpt2  <- read_csv("RESULTS/per_sample_predictions_gpt2-xl.csv", show_col_types = FALSE)
  behav_llama <- read_csv("RESULTS/behavioral_full_llama-3.1-8b.csv", show_col_types = FALSE)
  behav_gpt2  <- read_csv("RESULTS/behavioral_full_gpt2-xl.csv", show_col_types = FALSE)
  
  # Add Model Tags and Combine
  ci_llama$model <- "Llama 3.1 8B"
  ci_gpt2$model  <- "GPT-2 XL"
  ci_all <- bind_rows(ci_llama, ci_gpt2) %>%
    mutate(model = factor(model, levels = c("Llama 3.1 8B", "GPT-2 XL")))
  
  probe_llama$model <- "Llama 3.1 8B"
  probe_gpt2$model  <- "GPT-2 XL"
  probe_all <- bind_rows(probe_llama, probe_gpt2) %>%
    mutate(model = factor(model, levels = c("Llama 3.1 8B", "GPT-2 XL")))
  
  behav_llama$model <- "Llama 3.1 8B"
  behav_gpt2$model  <- "GPT-2 XL"
  behav_all <- bind_rows(behav_llama, behav_gpt2) %>%
    mutate(model = factor(model, levels = c("Llama 3.1 8B", "GPT-2 XL")))
  
  
  # %% Thesis Figure Setup
  TEXT_WIDTH_CM <- 15
  
  theme_thesis <- theme_minimal(base_size = 10) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = 10),
      plot.title = element_text(size = 11, face = "bold"),
      axis.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      legend.title = element_blank()
    )
  
  colours <- c(
    "explicit_leave" = "#F94A13",
    "baseline" = "#455A64",
    "implied_leave" = "#C9933A",
    "implied_cancel" = "#4A8C6F",
    "disengaged" = "#4A7FB5"
  )
  
  condition_labels <- c(
    "explicit_leave" = "Explicit Leave",
    "baseline"       = "Baseline",
    "implied_leave"  = "Implied Leave",
    "implied_cancel" = "Cancel",
    "disengaged"     = "Disengaged"
  )
  
  # %% Figure 0: Exploratory Graph (Dataset Splits), Stacked
  # Load the raw dataset
  raid_df <- read_csv("DATASET/RAID.csv", show_col_types = FALSE)
  
  # Count the samples per split and condition, and order them logically
  split_counts <- raid_df %>%
    count(split, condition) %>%
    mutate(
      split = factor(split, 
                     levels = c("train", "test", "generalisation"),
                     labels = c("Train", "Test", "Generalisation")),
      # Reverse the condition factor so the explicit/baseline conditions sit at the bottom of the stack
      condition = factor(condition, 
                         levels = rev(c("explicit_leave", "baseline", "implied_leave", "implied_cancel", "disengaged")))
    )
  
  # Generate the styled plot (Stacked Bar Chart)
  ggplot(split_counts, aes(x = split, y = n, fill = condition)) +
    # geom_col defaults to a stacked position. We add a thin dark border to separate the blocks cleanly.
    geom_col(colour = "grey20", linewidth = 0.3, width = 0.6) +
    # Place the numbers directly inside the middle of each colored block
    geom_text(aes(label = n), position = position_stack(vjust = 0.5), 
              size = 3.5, colour = "white", fontface = "bold") +
    # Use your exact thesis colors and labels
    scale_fill_manual(values = colours, labels = condition_labels) +
    labs(
      x = "Dataset Split", 
      y = "Total Number of Samples", 
      title = "RAID Dataset: Splits and Conditions"
    ) +
    theme_thesis +
    theme(
      axis.text.x = element_text(size = 11, face = "bold"),
      panel.grid.major.x = element_blank(), 
      legend.position = "right"
    )
  
  ggsave("fig_0_dataset_splits_stacked.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")
  
  # %% Figure 0b: Confounding Variables Balancing
  balance_long <- raid_df %>%
    select(structure, role_order, s3_variant) %>%
    pivot_longer(cols = everything(), names_to = "variable", values_to = "level") %>%
    count(variable, level) %>%
    group_by(variable) %>%
    mutate(
      # Calculate percentages (0.5 for role/s3, 0.25 for structure)
      percentage = n / sum(n),
      # Clean up the headers for the 3 facet panels
      var_label = case_when(
        variable == "structure" ~ "Sentence Structure",
        variable == "role_order" ~ "Role Order",
        variable == "s3_variant" ~ "Spatial Marker"
      ),
      var_label = factor(var_label, levels = c("Sentence Structure", "Role Order", "Spatial Marker"))
    ) %>% ungroup()
  
  balance_long <- balance_long %>%
    mutate(
      level_clean = case_when(
        level == "but" ~ "But",
        level == "whereas" ~ "Whereas",
        level == "while" ~ "While",
        level == "twosent" ~ "Two Sentences",
        level == "changer_first" ~ "Changer First",
        level == "stayer_first" ~ "Stayer First",
        level == "there" ~ "'There' Included",
        level == "nothere" ~ "No Marker"
      ),
      level_clean = factor(level_clean, levels = c("But", "Whereas", "While", "Two Sentences", 
                                                   "Changer First", "Stayer First", 
                                                   "'There' Included", "No Marker"))
    )
  
  # Generate the styled plot (3-Panel Grid)
  ggplot(balance_long, aes(x = level_clean, y = percentage, fill = variable)) +
    geom_col(width = 0.6, colour = "grey20", linewidth = 0.3) +
    # Add the bold percentage labels right on top of the bars
    geom_text(aes(label = scales::percent(percentage, accuracy = 1)), 
              vjust = -0.8, size = 3.5, fontface = "bold", colour = "grey20") +
    # Split the graph into 3 distinct panels!
    facet_wrap(~ var_label, scales = "free_x") +
    # Use percentages on the Y-axis and extend it slightly so the 50% text isn't cut off
    scale_y_continuous(labels = scales::percent_format(), limits = c(0, 0.6)) +
    # Use 3 of your distinct thesis colors for the different panels
    scale_fill_manual(values = c("structure" = "#455A64", "role_order" = "#C9933A", "s3_variant" = "#4A7FB5")) +
    labs(
      x = NULL,
      y = "Proportion of Dataset",
      title = "RAID Dataset: Balance of Structural Variables"
    ) +
    theme_thesis +
    theme(
      legend.position = "none", 
      axis.text.x = element_text(angle = 25, hjust = 1, size = 9),
      panel.grid.major.x = element_blank() # Remove vertical gridlines
    )
  
  ggsave("fig_0b_variable_balancing.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# PART 1: INTERNAL REPRESENTATIONS (PROBING)

# %% Figure 1: Training Poles
poles <- ci_all %>% filter(condition %in% c("explicit_leave", "baseline"), layer > 0)

ggplot(poles, aes(x = layer, y = value, colour = condition, fill = condition)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey50") +
  facet_wrap(~ model, scales = "free_x") +
  scale_colour_manual(values = colours, labels = condition_labels) +
  scale_fill_manual(values = colours, labels = condition_labels) +
  labs(x = "Layer", y = "Accuracy", title = "Training Poles") +
  ylim(0, 1.05) +
  theme_thesis

ggsave("fig_1_training_poles.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# %% Figure 2: Generalisation
gen <- ci_all %>% filter(condition %in% c("implied_leave", "implied_cancel", "disengaged"), layer > 0)

ggplot(gen, aes(x = layer, y = value, colour = condition, fill = condition)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey50") +
  facet_wrap(~ model, scales = "free_x") +
  scale_colour_manual(values = colours, labels = condition_labels) +
  scale_fill_manual(values = colours, labels = condition_labels) +
  labs(x = "Layer", y = "Accuracy", title = "Generalisation to Implied Conditions") +
  ylim(0, 1.05) +
  theme_thesis

ggsave("fig_2_generalisation.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# %% Figure 3: Linguistic Structure Analysis (Combined)
structure_colours <- c("but" = "#C9933A", "whereas" = "#7A4419", "while" = "#D9704A", "twosent" = "#455A64")
structure_labels <- c("but" = "But", "whereas" = "Whereas", "while" = "While", "twosent" = "Two Sentences")

struct_data_all <- probe_all %>%
  filter(condition == "implied_leave", layer > 0) %>%
  mutate(correct = as.integer(prediction == 1)) %>%
  group_by(model, layer, structure) %>%
  summarise(
    value = mean(correct),
    ci_low = quantile(replicate(1000, mean(sample(correct, replace = TRUE))), 0.025),
    ci_high = quantile(replicate(1000, mean(sample(correct, replace = TRUE))), 0.975),
    .groups = "drop"
  )

ggplot(struct_data_all, aes(x = layer, y = value, colour = structure, fill = structure)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.1, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey50") +
  facet_wrap(~ model, scales = "free_x") +
  scale_colour_manual(values = structure_colours, labels = structure_labels) +
  scale_fill_manual(values = structure_colours, labels = structure_labels) +
  labs(x = "Layer", y = "Accuracy", title = "Implied Leave by Sentence Structure") +
  coord_cartesian(ylim = c(0, 1.05)) +
  theme_thesis

ggsave("fig_3_structure_combined.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# %% Figure 4: Role Order Effect (Combined)
role_data_all <- probe_all %>%
  filter(condition == "implied_leave", layer > 0) %>%
  mutate(correct = as.integer(prediction == 1)) %>%
  group_by(model, layer, role_order) %>%
  summarise(
    value = mean(correct),
    ci_low = quantile(replicate(1000, mean(sample(correct, replace = TRUE))), 0.025),
    ci_high = quantile(replicate(1000, mean(sample(correct, replace = TRUE))), 0.975),
    .groups = "drop"
  )

ggplot(role_data_all, aes(x = layer, y = value, colour = role_order, fill = role_order)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.1, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey50") +
  facet_wrap(~ model, scales = "free_x") +
  scale_colour_manual(values = c("changer_first" = "#C9933A", "stayer_first" = "#455A64"),
                      labels = c("changer mentioned first", "stayer mentioned first")) +
  scale_fill_manual(values = c("changer_first" = "#C9933A", "stayer_first" = "#455A64"),
                    labels = c("changer mentioned first", "stayer mentioned first")) +
  labs(x = "Layer", y = "Accuracy", title = "Implied Leave by Role Order") +
  coord_cartesian(ylim = c(0, 1.05)) +
  theme_thesis

ggsave("fig_4_role_order_combined.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# %% Figure 5: Divergence with Bootstrapped CIs
compute_divergence_boot <- function(probe_data, model_name, n_boot = 1000) {
  probe_data <- probe_data %>% filter(layer > 0)
  layers <- sort(unique(probe_data$layer))
  
  leave_preds <- probe_data %>% filter(condition == "implied_leave") %>% mutate(correct = as.integer(prediction == 1))
  cancel_preds <- probe_data %>% filter(condition == "implied_cancel") %>% mutate(correct = as.integer(prediction == 0))
  
  map_dfr(layers, function(l) {
    leave_l <- leave_preds %>% filter(layer == l) %>% pull(correct)
    cancel_l <- cancel_preds %>% filter(layer == l) %>% pull(correct)
    
    boot_divs <- replicate(n_boot, { mean(sample(leave_l, replace = TRUE)) + mean(sample(cancel_l, replace = TRUE)) - 1 })
    
    tibble(layer = l, divergence = mean(leave_l) + mean(cancel_l) - 1,
           ci_low = quantile(boot_divs, 0.025), ci_high = quantile(boot_divs, 0.975), model = model_name)
  })
}

div_llama <- compute_divergence_boot(probe_llama, "Llama 3.1 8B")
div_gpt2  <- compute_divergence_boot(probe_gpt2, "GPT-2 XL")
div_all   <- bind_rows(div_llama, div_gpt2) %>% mutate(model = factor(model, levels = c("Llama 3.1 8B", "GPT-2 XL")))

ggplot(div_all, aes(x = layer, y = divergence)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), fill = "#455A64", alpha = 0.15) +
  geom_line(linewidth = 0.7, colour = "#455A64") +
  geom_hline(yintercept = 0, linetype = "dotted", colour = "grey50") +
  facet_wrap(~ model, scales = "free_x") +
  labs(x = "Layer", y = "Divergence", title = "Context Sensitivity, Implied Leave vs Implied Cancel") +
  coord_cartesian(ylim = c(-0.4, 0.4)) + 
  theme_thesis

ggsave("fig_5_divergence.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# %% Figure 11: Minimalist Decision Boundary Scatter Plot (Combined Models)
scatter_llama <- probe_llama %>% filter(layer %in% c(8, 16, 24, 32)) %>%
  filter(condition %in% c("explicit_leave", "implied_leave", "implied_cancel", "disengaged", "baseline")) %>%
  mutate(model = "Llama 3.1 8B", depth = case_when(layer == 8 ~ "25% Depth", layer == 16 ~ "50% Depth", layer == 24 ~ "75% Depth", layer == 32 ~ "100% Depth"))

scatter_gpt2 <- probe_gpt2 %>% filter(layer %in% c(12, 24, 36, 48)) %>%
  filter(condition %in% c("explicit_leave", "implied_leave", "implied_cancel", "disengaged", "baseline")) %>%
  mutate(model = "GPT-2 XL", depth = case_when(layer == 12 ~ "25% Depth", layer == 24 ~ "50% Depth", layer == 36 ~ "75% Depth", layer == 48 ~ "100% Depth"))

scatter_all <- bind_rows(scatter_llama, scatter_gpt2) %>%
  mutate(condition = factor(condition, levels = rev(c("baseline", "implied_cancel", "disengaged", "implied_leave", "explicit_leave")),
                            labels = rev(c("Baseline", "Implied Cancel", "Disengaged", "Implied Leave", "Explicit Leave"))),
         depth = factor(depth, levels = c("25% Depth", "50% Depth", "75% Depth", "100% Depth")),
         model = factor(model, levels = c("Llama 3.1 8B", "GPT-2 XL")))

ggplot(scatter_all, aes(x = prob_1_available, y = condition, colour = condition)) +
  geom_jitter(height = 0.25, alpha = 0.25, size = 0.7, stroke = 0) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "black", linewidth = 0.5) +
  facet_grid(model ~ depth) +
  scale_colour_manual(values = c("Explicit Leave" = "#F94A13", "Implied Leave" = "#C9933A", "Implied Cancel" = "#4A8C6F", "Disengaged" = "#4A7FB5", "Baseline" = "#455A64")) +
  scale_x_continuous(breaks = c(0, 0.5, 1), labels = c("0.0", "0.5", "1.0")) +
  labs(x = "Probe Confidence (Predicted Probability of 'Leave')", y = NULL, title = "Probe Decision Boundary Across Quartiles") +
  coord_cartesian(xlim = c(0, 1)) +
  theme_thesis +
  theme(legend.position = "none", panel.grid.major.y = element_blank(), axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 9, face = "bold"), panel.border = element_rect(colour = "grey80", fill = NA, linewidth = 0.5))

ggsave("fig_11_decision_boundary_combined.pdf", width = TEXT_WIDTH_CM, height = 10, units = "cm")


# PART 2: BEHAVIORAL OUTPUT & ALIGNMENT

# %% Figure 6: Behavioural Probability Distributions
behav_all$prob_diff <- behav_all$stayer_prob - behav_all$changer_prob

behav_plot <- behav_all %>% filter(condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  mutate(condition = factor(condition, levels = c("explicit_leave", "implied_leave", "disengaged"), labels = c("Explicit Leave", "Implied Leave", "Disengaged")))

ggplot(behav_plot, aes(x = condition, y = prob_diff, fill = condition)) +
  geom_violin(alpha = 0.6, colour = NA) +
  geom_boxplot(width = 0.15, outlier.size = 0.5, alpha = 0.8, fill = "white", colour = "grey20") + 
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
  facet_wrap(~ model) +
  scale_fill_manual(values = c("Explicit Leave" = "#F94A13", "Implied Leave" = "#C9933A", "Disengaged" = "#4A7FB5")) +
  labs(x = NULL, y = "P(Stayer) − P(Changer)", title = "Behavioural Output, Probability Difference") +
  coord_cartesian(ylim = c(-1, 1)) + 
  theme_thesis + theme(legend.position = "none", axis.text.x = element_text(angle = 15, hjust = 1))

ggsave("fig_6_behavioral_violin.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# %% Figure 7: Behavioural vs Probe Comparison
behav_summary <- behav_all %>% filter(condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  group_by(model, condition) %>% summarise(value = mean(correct, na.rm = TRUE), .groups = "drop") %>% mutate(measure = "Behavioural Output")

probe_final_llama <- probe_llama %>% filter(layer == max(layer), condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  mutate(correct = as.integer(prediction == 1)) %>% group_by(condition) %>% summarise(value = mean(correct), .groups = "drop") %>% mutate(model = "Llama 3.1 8B", measure = "Probe (Final Layer)")

probe_final_gpt2 <- probe_gpt2 %>% filter(layer == max(layer), condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  mutate(correct = as.integer(prediction == 1)) %>% group_by(condition) %>% summarise(value = mean(correct), .groups = "drop") %>% mutate(model = "GPT-2 XL", measure = "Probe (Final Layer)")

comparison <- bind_rows(behav_summary, probe_final_llama, probe_final_gpt2) %>%
  mutate(condition = factor(condition, levels = c("explicit_leave", "implied_leave", "disengaged"), labels = c("Explicit Leave", "Implied Leave", "Disengaged")),
         measure = factor(measure, levels = c("Probe (Final Layer)", "Behavioural Output")))

ggplot(comparison, aes(x = condition, y = value, fill = measure)) +
  geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.8) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey50") +
  facet_wrap(~ model) +
  scale_fill_manual(values = c("Probe (Final Layer)" = "#455A64", "Behavioural Output" = "#C9933A")) +
  labs(x = NULL, y = "Accuracy", title = "Representation vs. Generation, Final Layer Comparison") +
  coord_cartesian(ylim = c(0, 1.05)) + theme_thesis + theme(axis.text.x = element_text(angle = 15, hjust = 1), legend.position = "bottom", legend.title = element_blank())

ggsave("fig_7_probe_vs_behavioral.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# %% Figure 8: Probe vs Output Alignment Heatmap
make_alignment <- function(probe_data, behav_data, model_name) {
  final <- probe_data %>% filter(layer == max(layer), condition %in% c("explicit_leave", "implied_leave", "disengaged"))
  behav_sub <- behav_data %>% filter(condition %in% c("explicit_leave", "implied_leave", "disengaged"))
  tibble(condition = final$condition, probe_pred = case_when(final$prediction == 1 ~ "Correct", final$prediction == 0 ~ "Incorrect"),
         behav_pred = case_when(behav_sub$correct == TRUE ~ "Correct", behav_sub$correct == FALSE ~ "Incorrect"), model = model_name)
}

align_llama <- make_alignment(probe_llama, behav_llama, "Llama 3.1 8B")
align_gpt2  <- make_alignment(probe_gpt2, behav_gpt2, "GPT-2 XL")
align_counts <- bind_rows(align_llama, align_gpt2) %>%
  mutate(condition = factor(condition, levels = c("explicit_leave", "implied_leave", "disengaged"), labels = c("Explicit Leave", "Implied Leave", "Disengaged")),
         probe_pred = factor(probe_pred, levels = c("Incorrect", "Correct")), behav_pred = factor(behav_pred, levels = c("Correct", "Incorrect"))) %>%
  count(model, condition, probe_pred, behav_pred) %>% group_by(model, condition) %>% complete(probe_pred, behav_pred, fill = list(n = 0)) %>% mutate(pct = n / sum(n)) %>% ungroup()

ggplot(align_counts, aes(x = behav_pred, y = probe_pred, fill = pct)) +
  geom_tile(colour = "white", linewidth = 1) + geom_text(aes(label = scales::percent(pct, accuracy = 1)), size = 3.5) +
  facet_grid(model ~ condition) + scale_fill_gradient(low = "white", high = "#455A64", limits = c(0, 1)) +
  labs(x = "Behavioural Output", y = "Internal Probe Prediction", title = "Probe vs. Output Alignment", fill = "Proportion") +
  theme_thesis + theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right", panel.grid = element_blank())

ggsave("fig_8_alignment_heatmap.pdf", width = TEXT_WIDTH_CM, height = 10, units = "cm")


# %% Figure 9: Recency Bias
recency <- behav_all %>% filter(condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  mutate(second_mentioned = case_when(role_order == "changer_first" ~ "stayer", role_order == "stayer_first" ~ "changer"),
         picked_second = (predicted_role == second_mentioned)) %>%
  group_by(model, condition) %>% summarise(pct_second = mean(picked_second, na.rm = TRUE), .groups = "drop") %>%
  mutate(condition = factor(condition, levels = c("explicit_leave", "implied_leave", "disengaged"), labels = c("Explicit Leave", "Implied Leave", "Disengaged")))

ggplot(recency, aes(x = condition, y = pct_second, fill = condition)) +
  geom_col(alpha = 0.8, width = 0.6) + geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey50") +
  geom_text(aes(label = scales::percent(pct_second, accuracy = 0.1)), vjust = -0.8, size = 3.5) + facet_wrap(~ model) +
  scale_fill_manual(values = c("Explicit Leave" = "#F94A13", "Implied Leave" = "#C9933A", "Disengaged" = "#4A7FB5")) +
  labs(x = NULL, y = "Selection of Second-Mentioned Character (%)", title = "Behavioural Output, Recency Bias") +
  coord_cartesian(ylim = c(0, 1)) + theme_thesis + theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("fig_9_recency_bias.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# %% Figure 10: Name-Level Override 
name_override <- behav_all %>% filter(condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  mutate(first_role = ifelse(role_order == "changer_first", "changer", "stayer"), picked_first = (predicted_role == first_role))

override_by_name <- name_override %>% group_by(model, true_label) %>% summarise(override_rate = mean(picked_first, na.rm = TRUE), n = n(), .groups = "drop") %>% filter(n >= 10) %>%
  group_by(model) %>% arrange(desc(override_rate)) %>% mutate(rank = row_number(), total = n(), label = ifelse(rank <= 3 | rank > total - 3, as.character(true_label), "")) %>% ungroup()

ggplot(override_by_name, aes(x = model, y = override_rate)) +
  geom_boxplot(width = 0.4, alpha = 0.3, outlier.shape = NA, fill = "#455A64", colour = "grey30") +
  geom_jitter(width = 0.15, alpha = 0.6, size = 2, colour = "#455A64") +
  geom_text_repel(aes(label = label), size = 3, max.overlaps = 20, fontface = "italic", colour = "#F94A13") +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "grey50") +
  labs(x = NULL, y = "Selection Despite Being Mentioned First (%)", title = "Behavioural Output, Name-Level Recency Override") +
  coord_cartesian(ylim = c(0, 1.05)) + theme_thesis

ggsave("fig_10_name_override.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")


# PART 3: UNSUPERVISED COSINE SIMILARITY 

# %% Master Function to Process .npy Files (Corrected!)
compute_trajectory <- function(npy_file, csv_file, model_name) {
  hs_array <- np$load(npy_file)
  eval_data <- read_csv(csv_file, show_col_types = FALSE) 
  conditions <- eval_data$condition
  
  num_layers <- dim(hs_array)[2]
  cond_levels <- c("explicit_leave", "implied_leave", "disengaged", "implied_cancel", "baseline")
  
  trajectory_data <- data.frame()
  
  for (l in 1:num_layers) {
    layer_idx <- l 
    centroids <- list()
    for (cond in cond_levels) {
      row_indices <- which(conditions == cond)
      layer_data <- hs_array[row_indices, layer_idx, ]
      centroids[[cond]] <- colMeans(layer_data)
    }
    
    trajectory_data <- rbind(trajectory_data, data.frame(
      layer = l - 1, model = model_name,
      trick_to_leave = as.numeric(cosine(centroids[["implied_cancel"]], centroids[["explicit_leave"]])),
      trick_to_stay = as.numeric(cosine(centroids[["implied_cancel"]], centroids[["baseline"]])),
      implied_to_explicit = as.numeric(cosine(centroids[["implied_leave"]], centroids[["explicit_leave"]])),
      cancel_to_leave = as.numeric(cosine(centroids[["implied_cancel"]], centroids[["implied_leave"]]))
    ))
  }
  return(trajectory_data)
}

# Run extraction (Ensure file paths are correct!)
traj_llama <- compute_trajectory("RESULTS/HIDDENSTATES/hs_eval_llama-3.1-8b.npy", "RESULTS/behavioral_full_llama-3.1-8b.csv", "Llama 3.1 8B")
traj_gpt2  <- compute_trajectory("RESULTS/HIDDENSTATES/hs_eval_gpt2-xl.npy", "RESULTS/behavioral_full_gpt2-xl.csv", "GPT-2 XL")
traj_all <- bind_rows(traj_llama, traj_gpt2) %>% mutate(model = factor(model, levels = c("Llama 3.1 8B", "GPT-2 XL")))

# %% Figure 12: Cosine Similarity Matrix

cos_llama <- read_csv("RESULTS/cosine_matrix_llama-3.1-8b.csv", show_col_types = FALSE) %>% mutate(model = "Llama 3.1 8B")
cos_gpt2  <- read_csv("RESULTS/cosine_matrix_gpt2-xl.csv", show_col_types = FALSE) %>% mutate(model = "GPT-2 XL")

cos_all <- bind_rows(cos_llama, cos_gpt2) %>%
  mutate(
    cond1 = factor(cond1,
                   levels = c("explicit_leave", "implied_leave", "disengaged", "implied_cancel", "baseline"),
                   labels = c("Expl. Leave", "Impl. Leave", "Disengaged", "Impl. Cancel", "Baseline")),
    # Added rev() here to fix the visual diagonal!
    cond2 = factor(cond2,
                   levels = rev(c("explicit_leave", "implied_leave", "disengaged", "implied_cancel", "baseline")),
                   labels = rev(c("Expl. Leave", "Impl. Leave", "Disengaged", "Impl. Cancel", "Baseline"))),
    model = factor(model, levels = c("Llama 3.1 8B", "GPT-2 XL"))
  )

# Calculate the actual minimum similarity so our color scale doesn't clip!
min_sim <- min(cos_all$similarity, na.rm = TRUE)

# Calculate the middle point of your data to decide when to flip the text color
mid_sim <- min_sim + ((1.0 - min_sim) / 2)

ggplot(cos_all, aes(x = cond1, y = cond2, fill = round(similarity, 3))) +
  geom_tile(colour = "white", linewidth = 1) +
  # THE TEXT FIX: Dynamically change text color based on how dark the background is!
  geom_text(aes(label = sprintf("%.3f", similarity), 
                colour = similarity > mid_sim), size = 3) +
  # Map the TRUE/FALSE from above to White/Black
  scale_colour_manual(values = c("TRUE" = "white", "FALSE" = "black"), guide = "none") +
  facet_wrap(~ model) +
  # THE COLOR FIX: White to your Dark Grey/Blue for a much cleaner gradient
  scale_fill_gradient(low = "white", high = "#455A64", limits = c(min_sim, 1.0)) +
  labs(x = NULL, y = NULL,
       title = "Cosine Similarity at Final Layer") +
  theme_thesis +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "right", panel.grid = element_blank())

ggsave("fig_12_cosine_heatmap.pdf", width = TEXT_WIDTH_CM, height = 8, units = "cm")

# BEHAVIOURAL STATISTICS - VIOLIN PLOT NUMBERS

# Probability difference summary statistics for violin plot
behav_violin_stats <- behav_all %>%
  filter(condition %in% c("explicit_leave", "implied_leave", "disengaged")) %>%
  mutate(prob_diff = stayer_prob - changer_prob) %>%
  group_by(model, condition) %>%
  summarise(
    n = n(),
    mean_diff = mean(prob_diff),
    median_diff = median(prob_diff),
    sd_diff = sd(prob_diff),
    pct_positive = mean(prob_diff > 0),
    # One sample t-test: is mean diff different from 0?
    t_stat = t.test(prob_diff, mu = 0)$statistic,
    p_value = t.test(prob_diff, mu = 0)$p.value,
    ci_low = t.test(prob_diff, mu = 0)$conf.int[1],
    ci_high = t.test(prob_diff, mu = 0)$conf.int[2],
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

print(behav_violin_stats)
write_csv(behav_violin_stats, "RESULTS/stats_behavioral_violin.csv")

# APPENDIX: LDA VISUALISATION OF HIDDEN STATES

plot_pca_lda <- function(npy_file, csv_file, model_name, target_layers, n_pca_comps = 50) {
  
  hs <- np$load(npy_file)
  meta <- read_csv(csv_file, show_col_types = FALSE)
  
  plots <- list()
  
  for (layer in target_layers) {
    # Extract hidden states for the target layer
    layer_data <- as.data.frame(hs[, layer + 1, ])
    
    # 1. Run PCA first to compress dimensionality and remove collinearity
    pca_result <- prcomp(layer_data, center = TRUE, scale. = TRUE)
    
    # Take the top N principal components (safeguard against taking more than available)
    n_comps <- min(n_pca_comps, nrow(layer_data) - 1, ncol(layer_data))
    pca_df <- as.data.frame(pca_result$x[, 1:n_comps])
    
    # Add the condition labels back to our compressed data
    pca_df$condition <- meta$condition
    
    # 2. Run LDA on the compressed Principal Components
    lda_result <- lda(condition ~ ., data = pca_df)
    lda_scores <- predict(lda_result)$x
    
    # 3. Prepare for plotting
    lda_df <- data.frame(
      LD1 = lda_scores[, 1],
      LD2 = lda_scores[, 2],
      condition = meta$condition
    ) %>%
      mutate(condition = factor(condition,
                                levels = c("explicit_leave", "baseline",
                                           "implied_leave", "implied_cancel",
                                           "disengaged")))
    
    # 4. Generate the plot using your custom thesis theme
    p <- ggplot(lda_df, aes(x = LD1, y = LD2, colour = condition)) +
      geom_point(alpha = 0.4, size = 0.8) +
      stat_ellipse(level = 0.95, linewidth = 0.8) +
      scale_colour_manual(values = colours, labels = condition_labels) +
      labs(
        x = "LD1",
        y = "LD2",
        title = sprintf("Layer %d", layer)
      ) +
      theme_thesis +
      theme(legend.position = "none") # Hide individual legends to collect them later
    
    plots[[as.character(layer)]] <- p
  }
  
  return(plots)
}

# --- GENERATE THE PLOTS ---

# LLaMA (Layers to visualize)
llama_layers <- c(1, 8, 16, 24, 32)
llama_lda <- plot_pca_lda(
  "RESULTS/HIDDENSTATES/hs_eval_llama-3.1-8b.npy",
  "RESULTS/behavioral_full_llama-3.1-8b.csv",
  "Llama 3.1 8B",
  llama_layers,
  n_pca_comps = 50
)

# GPT-2 XL (Layers to visualize)
gpt_layers <- c(1, 12, 24, 36, 48)
gpt_lda <- plot_pca_lda(
  "RESULTS/HIDDENSTATES/hs_eval_gpt2-xl.npy",
  "RESULTS/behavioral_full_gpt2-xl.csv",
  "GPT-2 XL",
  gpt_layers,
  n_pca_comps = 50
)

# COMBINE WITH PATCHWORK

llama_lda_combined <- 
  llama_lda[["1"]] + llama_lda[["8"]] + llama_lda[["16"]] + 
  llama_lda[["24"]] + llama_lda[["32"]] +
  plot_layout(ncol = 5, guides = "collect") &
  theme(legend.position = "bottom")

gpt_lda_combined <- 
  gpt_lda[["1"]] + gpt_lda[["12"]] + gpt_lda[["24"]] + 
  gpt_lda[["36"]] + gpt_lda[["48"]] +
  plot_layout(ncol = 5, guides = "collect") &
  theme(legend.position = "bottom")

# Save them!
ggsave("fig_app_lda_llama.pdf", llama_lda_combined,
       width = TEXT_WIDTH_CM * 1.5, height = 7, units = "cm")
ggsave("fig_app_lda_gpt2.pdf", gpt_lda_combined,
       width = TEXT_WIDTH_CM * 1.5, height = 7, units = "cm")


