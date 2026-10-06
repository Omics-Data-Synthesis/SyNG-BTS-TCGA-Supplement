#!/usr/bin/env Rscript

# RNA-seq manuscript figure using the unified Stage 05 outputs.
# Panel A summarizes selected configurations; Panel B shows five UMAP panels.
# One-group settings never use AE-head augmentation.

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(file_arg)) sub("^--file=", "", file_arg[[1]]) else "."
SCRIPT_DIR <- dirname(normalizePath(script_path))
ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
RESULTS_DIR <- file.path(ROOT, "results")
OUT_DIR <- file.path(ROOT, "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

blue <- "#2878B5"; orange <- "#E6863B"; teal <- "#007A4D"
guide_grey <- "#A8A8A8"; text_grey <- "#53636D"

selection <- data.frame(
  Cohort = c("LAML", "PAAD", "READ", "COAD", "SKCM"),
  dataset_id = c("RNA_LAML", "RNA_PAAD", "RNA_READ",
                 "RNA_COAD_tumor_status", "RNA_SKCM_breslow_thickness_at_diagnosis"),
  Setup = c("One-group VAE", "One-group VAE", "One-group VAE",
            "Two-group cVAE", "Two-group cVAE"),
  norm = c("raw", "raw", "raw", "TC", "TC"),
  model = c("VAE1-100", "VAE1-100", "VAE1-100", "CVAE1-100", "CVAE1-100"),
  off_aug = rep("none", 5),
  y = 5:1,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

source(file.path(SCRIPT_DIR, "fidelity_data.R"), local = TRUE)
fidelity <- read_fidelity(ROOT) %>%
  filter(modality == "rnaseq")
umap_data <- read.csv(file.path(RESULTS_DIR, "umap.csv"), stringsAsFactors = FALSE) %>%
  filter(modality == "rnaseq")

selected <- inner_join(fidelity, selection,
                       by = c("dataset_id", "norm", "model", "off_aug"))
if (nrow(selected) != 25) {
  stop(sprintf("Expected 25 selected fidelity rows (5 cohorts x 5 draws), found %d.", nrow(selected)))
}

long <- selected %>%
  mutate(ARI = source_ari) %>%
  select(Cohort, y, draw, ARI, medae_mean, medae_sd, medae_cv) %>%
  pivot_longer(c(ARI, medae_mean, medae_sd, medae_cv), names_to = "metric", values_to = "value") %>%
  group_by(Cohort, y, metric) %>%
  summarise(mean = mean(value, na.rm = TRUE), low = min(value, na.rm = TRUE),
            high = max(value, na.rm = TRUE), .groups = "drop")

metric_labels <- c(ARI = "Signed source ARI", medae_mean = "MedAE mean",
                   medae_sd = "MedAE SD", medae_cv = "MedAE CV")
metric_order <- names(metric_labels)

# Use one scale per metric, fixed across all five cohorts.
spec <- long %>% group_by(metric) %>% summarise(maximum = max(high, na.rm = TRUE), .groups = "drop") %>%
  mutate(limit = case_when(
    metric == "ARI" ~ pmax(.01, ceiling(maximum * 100) / 100),
    metric == "medae_mean" ~ pmax(.1, ceiling(maximum * 10) / 10),
    metric == "medae_sd" ~ pmax(.5, ceiling(maximum * 2) / 2),
    TRUE ~ pmax(.1, ceiling(maximum * 10) / 10)
  ))
spec <- spec[match(metric_order, spec$metric), ]

metric_panel <- function(key, limit) {
  d <- filter(long, metric == key)
  lower <- if (key == "ARI") min(-.01, floor(min(d$low) * 100) / 100) else 0
  br <- if (key == "ARI") c(lower, 0, limit) else c(0, limit / 2, limit)
  padding <- .08 * (limit - lower)
  ggplot(d, aes(mean, y)) +
    {if (key == "ARI") geom_vline(xintercept = 0, color = guide_grey, linetype = "dashed")} +
    geom_segment(aes(x = lower, xend = limit, yend = y), color = guide_grey, linewidth = .6) +
    geom_errorbar(aes(xmin = low, xmax = high), orientation = "y",
                  width = 0, color = teal, linewidth = .85) +
    geom_point(color = teal, size = 2.7) +
    scale_x_continuous(limits = c(lower - padding, limit + padding), breaks = br,
                       labels = format(br, trim = TRUE, scientific = FALSE),
                       expand = expansion(mult = c(0, 0))) +
    scale_y_continuous(limits = c(.5, 5.5), breaks = NULL) +
    labs(x = NULL, y = NULL) +
    theme_classic(base_size = 12) +
    theme(axis.text.x = element_text(size = 10), axis.ticks.y = element_blank(),
          axis.line.y = element_blank(), plot.margin = margin(3, 8, 3, 8))
}

p_labels <- ggplot(selection, aes(y = y)) +
  geom_text(aes(x = .05, label = Cohort), hjust = 0, size = 4.2) +
  geom_text(aes(x = .48, label = Setup), hjust = .5, size = 4.0) +
  geom_text(aes(x = .88, label = norm), hjust = .5, size = 4.1) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(.5, 5.5)) +
  theme_void(base_size = 8) +
  theme(plot.margin = margin(3, 5, 20, 0))

metric_plots <- Map(metric_panel, spec$metric, spec$limit)
p_label_header <- ggplot() +
  annotate("text", x = .05, y = .5, label = "Cohort", hjust = 0, size = 4.1, fontface = "bold") +
  annotate("text", x = .48, y = .5, label = "Setup", hjust = .5, size = 4.1, fontface = "bold") +
  annotate("text", x = .88, y = .5, label = "Normalization", hjust = .5,
           size = 4.1, fontface = "bold") +
  xlim(0, 1) + ylim(0, 1) + theme_void() + theme(plot.margin = margin(0, 5, 0, 0))
metric_headers <- lapply(unname(metric_labels), function(label) {
  ggplot() + annotate("text", x = .5, y = .5, label = label,
                      size = 4.1, fontface = "bold") +
    xlim(0, 1) + ylim(0, 1) + theme_void() + theme(plot.margin = margin(0, 8, 0, 8))
})
p_headers <- wrap_plots(c(list(p_label_header), metric_headers), nrow = 1,
                        widths = c(2.35, rep(1, 4)))
p_body <- wrap_plots(c(list(p_labels), metric_plots), nrow = 1,
                     widths = c(2.35, rep(1, 4)))
p_summary <- p_headers / p_body + plot_layout(heights = c(.11, 1))
p_summary_title <- ggplot() +
  annotate("text", x = .5, y = .5,
           label = "ARI: zero = chance baseline; negative = below chance. Lower MedAE is better.", size = 3.8) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  labs(title = "A  RNA-seq augmentation configurations and fidelity",
       subtitle = "Signed source ARI and marker-level MedAE; circles are means and green lines span five draws") +
  theme_void() +
  theme(plot.title = element_text(size = 17, face = "bold", hjust = 0),
        plot.subtitle = element_text(size = 12.5, color = text_grey, hjust = 0,
                                     margin = margin(t = 3)))
p_a <- p_summary_title / p_summary + plot_layout(heights = c(.21, 1))

# UMAP panels use the exact same selected settings as Panel A.
u <- inner_join(umap_data, selection,
                by = c("dataset_id", "norm", "model", "off_aug")) %>%
  mutate(
    Source = factor(recode(datatype, Real = "Empirical", Generated = "Generated"),
                    levels = c("Empirical", "Generated")),
    Group = case_when(
      Setup == "One-group VAE" ~ "One group",
      is.na(groups) ~ "Unknown",
      TRUE ~ paste("Group", as.integer(groups))
    ),
    Group = factor(Group, levels = c("One group", "Group 0", "Group 1", "Unknown"))
  )
if (nrow(u) == 0) stop("No UMAP rows matched the selected configurations.")

umap_panel <- function(cohort) {
  d <- filter(u, Cohort == cohort)
  subtitle <- unique(paste0(ifelse(grepl("^C", unique(d$model)), "cVAE-1:100", "VAE-1:100"),
                            " - ", unique(d$norm)))
  ggplot(d, aes(UMAP1, UMAP2, color = Source, shape = Group)) +
    geom_point(size = 2.05, alpha = .90, stroke = .20) +
    scale_color_manual(values = c(Empirical = blue, Generated = orange), drop = FALSE) +
    scale_shape_manual(
      values = c(`One group` = 16, `Group 0` = 16, `Group 1` = 17),
      limits = c("One group", "Group 0", "Group 1"),
      drop = FALSE,
      name = "Group"
    ) + guides(color = "none", shape = "none") +
    labs(title = cohort, subtitle = subtitle, x = "UMAP 1", y = "UMAP 2") +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(size = 14.5, face = "bold", hjust = 0),
          plot.subtitle = element_text(size = 11.5, color = text_grey, hjust = 0),
          axis.text = element_blank(), axis.ticks = element_blank(),
          panel.grid = element_line(color = "#D5D5D5", linewidth = .3),
          panel.border = element_rect(color = "#909090", linewidth = .45),
          aspect.ratio = .72, plot.margin = margin(3, 10, 3, 3))
}

umap_plots <- lapply(selection$Cohort, umap_panel)
p_one <- wrap_plots(umap_plots[1:3], nrow = 1)
p_two <- plot_spacer() + umap_plots[[4]] + umap_plots[[5]] + plot_spacer() +
  plot_layout(nrow = 1, widths = c(.45, 1, 1, .45))
p_one_label <- ggplot() + annotate("text", x = 0, y = .5, label = "One-group VAE cohorts",
                                  hjust = 0, size = 4.7, color = text_grey,
                                  fontface = "bold") +
  xlim(0, 1) + ylim(0, 1) + theme_void()
p_two_label <- ggplot() + annotate("text", x = 0, y = .5, label = "Two-group cVAE cohorts",
                                  hjust = 0, size = 4.7, color = text_grey,
                                  fontface = "bold") +
  xlim(0, 1) + ylim(0, 1) + theme_void()

legend_source <- data.frame(x = c(.17, .39), y = .5,
                            label = c("Empirical", "Generated"),
                            Source = factor(c("Empirical", "Generated"),
                                            levels = c("Empirical", "Generated")))
legend_group <- data.frame(x = c(.68, .88), y = .5,
                           label = c("Group 0", "Group 1"),
                           Group = factor(c("Group 0", "Group 1"),
                                          levels = c("Group 0", "Group 1")))
p_legend <- ggplot() +
  annotate("text", x = .02, y = .5, label = "Source", hjust = 0, size = 4.1) +
  geom_point(data = legend_source, aes(x, y, color = Source), size = 2.1) +
  geom_text(data = legend_source, aes(x + .025, y, label = label), hjust = 0, size = 3.9) +
  annotate("text", x = .53, y = .5, label = "Group", hjust = 0, size = 4.1) +
  geom_point(data = legend_group, aes(x, y, shape = Group), size = 2.1, color = "black") +
  geom_text(data = legend_group, aes(x + .025, y, label = label), hjust = 0, size = 3.9) +
  scale_color_manual(values = c(Empirical = blue, Generated = orange)) +
  scale_shape_manual(values = c(`Group 0` = 1, `Group 1` = 2)) +
  guides(color = "none", shape = "none") +
  xlim(0, 1) + ylim(0, 1) + theme_void()

p_b_grid <- wrap_plots(p_one_label, p_one, p_two_label, p_two, p_legend,
                       ncol = 1, heights = c(.08, 1, .08, 1, .12))
p_b_title <- ggplot() +
  annotate("text", x = -.047, y = 0.5,
           label = "B  UMAP embeddings of empirical and generated RNA-seq samples",
           hjust = 0, size = 6.0, fontface = "bold", color = "black") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") + theme_void()
p_b_note <- ggplot() +
  annotate("text", x = 0, y = 0.5,
           label = "One-group samples are shown as circles. UMAP coordinates are cohort-specific and are not comparable across panels.",
           hjust = 0, size = 3.8, color = text_grey) +
  xlim(0, 1) + ylim(0, 1) + theme_void()
p_b <- wrap_plots(p_b_title, p_b_grid, p_b_note, ncol = 1,
                  heights = c(0.06, 1, 0.05))

fig <- wrap_plots(list(p_a, p_b), ncol = 1, heights = c(.58, 1.62)) &
  theme(text = element_text(family = "sans"))

out <- file.path(OUT_DIR, "RNA_selected_fidelity_UMAP.png")
ggsave(out, fig, width = 11.5, height = 12.5, units = "in", dpi = 300, bg = "white")
cat(out, "\n")
