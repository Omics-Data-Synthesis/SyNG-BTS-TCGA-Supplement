#!/usr/bin/env Rscript

# Generic miRNA cohort figure using the unified Stage 05 outputs.
# Run from any directory:
#   Rscript workflow/07_figures/fidelity_miRNA_code.R [dataset_id] [UMAP norm] [UMAP model]

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(file_arg)) sub("^--file=", "", file_arg[[1]]) else "."
SCRIPT_DIR <- dirname(normalizePath(script_path))
ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
RESULTS_DIR <- file.path(ROOT, "results")
OUT_DIR <- file.path(ROOT, "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

args <- commandArgs(trailingOnly = TRUE)
dataset <- if (length(args) >= 1) args[[1]] else "STAD_prospective_collection"
umap_norm <- if (length(args) >= 2) args[[2]] else "TC"
umap_model <- if (length(args) >= 3) args[[3]] else "CVAE1-10"

norms <- c("raw", "TC", "DESeq")
models <- c("CVAE1-5", "CVAE1-10", "CVAE1-50")
model_ratio <- c("CVAE1-5" = "1:5", "CVAE1-10" = "1:10", "CVAE1-50" = "1:50")
if (!umap_norm %in% norms) stop("UMAP norm must be raw, TC, or DESeq.")
if (!umap_model %in% models) stop("UMAP model must be CVAE1-5, CVAE1-10, or CVAE1-50.")

blue <- "#2878B5"; orange <- "#E6863B"; teal <- "#007A4D"
guide_grey <- "#A8A8A8"; text_grey <- "#53636D"

fidelity_all <- read.csv(file.path(RESULTS_DIR, "fidelity.csv"), stringsAsFactors = FALSE)
umap_all <- read.csv(file.path(RESULTS_DIR, "umap.csv"), stringsAsFactors = FALSE)

fidelity <- fidelity_all %>%
  filter(modality == "mirna", dataset_id == dataset, off_aug == "none",
         norm %in% norms, model %in% models)
if (nrow(fidelity) != 45) {
  stop(sprintf("Expected 45 miRNA fidelity rows (9 configurations x 5 draws), found %d.",
               nrow(fidelity)))
}

# Stage 05 already generated the representative, group-matched UMAP coordinates.
u <- umap_all %>%
  filter(modality == "mirna", dataset_id == dataset, off_aug == "none",
         norm == umap_norm, model == umap_model) %>%
  mutate(Source = factor(recode(datatype, Real = "Empirical", Generated = "Generated"),
                         levels = c("Empirical", "Generated")),
         Group = factor(as.character(groups)))
if (nrow(u) == 0) stop("No matching UMAP rows found in results/umap.csv.")
shape_values <- setNames(c(16, 17, 15, 18, 8, 3)[seq_along(levels(u$Group))], levels(u$Group))

p_umap <- ggplot(u, aes(UMAP1, UMAP2, color = Source, shape = Group)) +
  geom_point(size = 2.60, alpha = .95, stroke = .20) +
  scale_color_manual(values = c(Empirical = blue, Generated = orange)) +
  scale_shape_manual(values = shape_values) +
  labs(title = paste0("A  ", sub("_.*$", "", dataset), " miRNA samples in UMAP space"),
       subtitle = sprintf("Empirical and generated data - cVAE-%s, %s", model_ratio[[umap_model]], umap_norm),
       x = "UMAP 1", y = "UMAP 2") +
  theme_classic(base_size = 12) +
  theme(plot.title = element_text(size = 15, face = "bold", hjust = 0),
        plot.subtitle = element_text(size = 12, color = text_grey, hjust = 0),
        axis.text = element_blank(), axis.ticks = element_blank(),
        panel.grid.major = element_line(color = "#D5D5D5", linewidth = .35),
        aspect.ratio = .50, legend.position = "bottom", legend.box = "horizontal",
        legend.text = element_text(size = 10.5), legend.title = element_text(size = 10.5),
        legend.margin = margin(t = 0), legend.key.width = unit(.30, "inches"))

values_long <- fidelity %>%
  select(norm, model, draw, cccpcc, ccc_log10_pvalue, ccc_meandiff,
         medae_mean, medae_sd, medae_cv) %>%
  pivot_longer(-c(norm, model, draw), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
                         cccpcc = "CCC-PCC",
                         ccc_log10_pvalue = "CCC-log10(p)",
                         ccc_meandiff = "CCC mean",
                         medae_mean = "MedAE mean",
                         medae_sd = "MedAE sd",
                         medae_cv = "MedAE cv"))

row_order <- data.frame(norm = rep(c("raw", "TC", "DESeq"), each = 3),
                        model = rep(models, 3), loss = rep(c("1:5", "1:10", "1:50"), 3), y = 9:1)
vals <- values_long %>% group_by(norm, model, metric) %>%
  summarise(mean = mean(value, na.rm = TRUE), low = min(value, na.rm = TRUE),
            high = max(value, na.rm = TRUE), .groups = "drop") %>%
  left_join(row_order, by = c("norm", "model"))

# Limits are derived once per metric, then held fixed across all nine rows.
spec <- vals %>% group_by(metric) %>% summarise(maximum = max(high, na.rm = TRUE), .groups = "drop") %>%
  mutate(limit = case_when(grepl("^CCC", metric) ~ 1,
                           TRUE ~ pmax(.05, ceiling(maximum * 20) / 20)))
metric_order <- c("CCC-PCC", "CCC-log10(p)", "CCC mean", "MedAE mean", "MedAE sd", "MedAE cv")
spec <- spec[match(metric_order, spec$metric), ]

metric_plot <- function(key, limit) {
  d <- filter(vals, metric == key); br <- c(0, limit / 2, limit)
  p <- ggplot(d, aes(mean, y)) +
    geom_segment(aes(x = 0, xend = limit, yend = y), color = guide_grey, linewidth = .55) +
    geom_errorbar(aes(xmin = pmax(low, 0), xmax = pmin(high, limit)), orientation = "y",
                  width = 0, linewidth = .8, color = teal) +
    geom_point(color = teal, size = 2.5) +
    scale_x_continuous(limits = c(-.08 * limit, 1.08 * limit), breaks = br,
                       labels = format(br, trim = TRUE, scientific = FALSE),
                       oob = scales::squish, expand = expansion(mult = c(0, 0))) +
    scale_y_continuous(limits = c(.5, 9.5), breaks = NULL) +
    labs(title = recode(key, `MedAE sd` = "MedAE SD", `MedAE cv` = "MedAE CV"), x = NULL, y = NULL) +
    theme_classic(base_size = 12) +
    theme(plot.title = element_text(size = 11, face = "bold", hjust = .5, margin = margin(b = 7)),
          axis.text.x = element_text(size = 9.5), axis.ticks.y = element_blank(),
          axis.line.y = element_blank(), plot.margin = margin(3, 6, 3, 6))
  p
}

p_labels <- ggplot(row_order, aes(y = y)) +
  annotate("text", x = .22, y = c(8, 5, 2), label = c("raw", "TC", "DESeq"),
           size = 4.3, hjust = .5) +
  geom_text(aes(x = .86, label = loss), size = 4.2, hjust = .5) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(.5, 9.8), breaks = NULL, expand = c(0, 0)) +
  annotate("text", x = .22, y = 9.65, label = "Normalization", size = 4.3,
           fontface = "bold", hjust = .5) +
  annotate("text", x = .86, y = 9.65, label = "Loss ratio", size = 4.3,
           fontface = "bold", hjust = .5) +
  coord_cartesian(clip = "off") +
  theme_void(base_size = 12) +
  theme(plot.margin = margin(3, 5, 20, 0))

plots <- Map(metric_plot, spec$metric, spec$limit)
p_table <- wrap_plots(c(list(p_labels), plots), nrow = 1, widths = c(1.55, rep(1, 6)))
p_head <- ggplot() +
  annotate("segment", x = .25, xend = .57, y = .44, yend = .44,
           arrow = arrow(length = unit(.08, "inches"))) +
  annotate("text", x = .41, y = .76, label = "Higher is better", size = 4.7) +
  annotate("segment", x = .95, xend = .63, y = .44, yend = .44,
           arrow = arrow(length = unit(.08, "inches"))) +
  annotate("text", x = .79, y = .76, label = "Lower is better", size = 4.7) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  labs(title = "B  Fidelity across cVAE configurations") + theme_void() +
  theme(plot.title = element_text(size = 15, face = "bold", hjust = 0))

p_b <- p_head / p_table + plot_layout(heights = c(.13, 1))

umap_row <- p_umap
fig <- umap_row / p_b +
  plot_layout(heights = c(1.08, 1.20)) &
  theme(text = element_text(family = "sans"))

safe <- gsub("[^A-Za-z0-9_-]", "_", dataset)
out <- file.path(OUT_DIR, paste0(safe, "_miRNA_fidelity.png"))
png(out, width = 12.5, height = 10.2, units = "in", res = 300, bg = "white")
print(fig)
grid::grid.lines(
  x = grid::unit(c(.606, .606), "npc"),
  y = grid::unit(c(.055, .465), "npc"),
  gp = grid::gpar(col = "#808080", lwd = 1.4)
)
dev.off()
cat(out, "\n")
