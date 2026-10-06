#!/usr/bin/env Rscript

# Figure from the unified Stage 05 fidelity output.
# Signed source ARI; points are means of unrounded per-draw values.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(file_arg)) sub("^--file=", "", file_arg[[1]]) else "."
SCRIPT_DIR <- dirname(normalizePath(script_path))
ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
RESULTS_DIR <- file.path(ROOT, "results")
OUT_DIR <- file.path(ROOT, "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

blue <- "#180F80"
orange <- "#F28E00"
grey <- "#9B9B9B"
grid_grey <- "#C9C9C9"

source(file.path(SCRIPT_DIR, "fidelity_data.R"), local = TRUE)
fidelity <- read_fidelity(ROOT)

mirna <- fidelity %>%
  filter(modality == "mirna", norm == "raw", model == "CVAE1-10", off_aug == "none") %>%
  mutate(Cohort = sub("_.*$", "", dataset_id)) %>%
  group_by(Cohort) %>%
  summarise(mean = mean(source_ari), .groups = "drop") %>%
  arrange(mean) %>%
  mutate(Cohort = factor(Cohort, levels = Cohort))

# Selected RNA configurations. Edit this table when additional finalized
# configurations become available in fidelity.csv.
rna_selection <- data.frame(
  Cohort = c("LAML", "PAAD", "READ", "COAD", "SKCM"),
  dataset_id = c("RNA_LAML", "RNA_PAAD", "RNA_READ",
                 "RNA_COAD_tumor_status", "RNA_SKCM_breslow_thickness_at_diagnosis"),
  norm = c("raw", "raw", "raw", "TC", "TC"),
  model = c("VAE1-100", "VAE1-100", "VAE1-100", "CVAE1-100", "CVAE1-100"),
  off_aug = rep("none", 5),
  stringsAsFactors = FALSE
)

rna <- inner_join(filter(fidelity, modality == "rnaseq"), rna_selection,
                  by = c("dataset_id", "norm", "model", "off_aug")) %>%
  group_by(Cohort) %>%
  summarise(mean = mean(source_ari), .groups = "drop") %>%
  arrange(mean) %>%
  mutate(Cohort = factor(Cohort, levels = Cohort))

if (nrow(mirna) != 24) stop("Expected 24 miRNA cohorts for raw / CVAE1-10.")
if (nrow(rna) != 5) stop("Expected five selected RNA cohorts.")

y_top <- max(.12, ceiling(max(c(mirna$mean, rna$mean)) * 100) / 100 + .01)
y_bottom <- min(-.01, floor(min(c(mirna$mean, rna$mean)) * 100) / 100 - .01)
y_breaks <- sort(unique(c(y_bottom, seq(0, y_top, by = .05))))

panel_theme <- theme_classic(base_size = 11) +
  theme(
    axis.title = element_text(size = 11),
    axis.text.x = element_text(size = 9.5, angle = 58, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 9.5),
    axis.ticks.x = element_blank(),
    panel.grid.major.y = element_line(color = grid_grey, linewidth = .35,
                                      linetype = "dotted"),
    plot.margin = margin(0, 8, 5, 5)
  )

lollipop <- function(dat, color, show_y = TRUE) {
  ggplot(dat, aes(Cohort, mean)) +
    geom_hline(yintercept = 0, color = "#555555", linewidth = .4) +
    geom_segment(aes(xend = Cohort, y = 0, yend = mean), color = grey, linewidth = .6) +
    geom_point(color = color, size = 3.4) +
    geom_text(aes(label = ari_label(mean)), color = color,
              size = 3.0, vjust = -.9) +
    scale_y_continuous(limits = c(y_bottom, y_top), breaks = y_breaks,
                       labels = sprintf("%.2f", y_breaks),
                       expand = expansion(mult = c(0, .06))) +
    labs(x = NULL, y = if (show_y) "Signed source ARI" else NULL) +
    panel_theme +
    theme(
      axis.text.y = if (show_y) element_text(size = 9.5) else element_blank(),
      axis.ticks.y = if (show_y) element_line() else element_blank(),
      axis.line.y = if (show_y) element_line() else element_blank()
    )
}

p_a <- lollipop(mirna, blue, TRUE)
p_b <- lollipop(rna, orange, FALSE)

# Put both panel headings in one fixed-height row so their tops and baselines
# remain exactly aligned even though panel A has a y axis and panel B does not.
panel_title <- function(label) {
  ggplot() +
    annotate("text", x = 0, y = 0.5, label = label,
             hjust = 0, vjust = 0.5, size = 4.6, fontface = "bold") +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_void() +
    theme(plot.margin = margin(0, 8, 0, 5))
}

title_row <- panel_title("A  miRNA-seq - raw - cVAE-1:10") +
  panel_title("B  RNA-seq - selected configurations") +
  plot_layout(widths = c(1.72, 1))

body_row <- p_a + p_b + plot_layout(widths = c(1.72, 1))

caption <- paste0(
  "Signed ARI: zero is the chance-adjusted agreement baseline; negative values indicate below-chance agreement. ",
  "Points and labels show means across 5 draws.\nRNA: one-group cohorts use raw VAE-1:100; ",
  "two-group cohorts use TC cVAE-1:100. No selected setting uses AE-head augmentation."
)

fig <- title_row / body_row +
  plot_layout(heights = c(0.075, 1)) +
  plot_annotation(
    title = "Data-source separation across TCGA cohorts",
    caption = caption,
    theme = theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0, margin = margin(b = 10)),
      plot.caption = element_text(size = 9.5, color = "#555555", hjust = 0, margin = margin(t = 9))
    )
  ) & theme(text = element_text(family = "sans"))

out <- file.path(OUT_DIR, "data_source_separation_miRNA_RNA.png")
ggsave(out, fig, width = 12.8, height = 6.8, units = "in", dpi = 300, bg = "white")
cat(out, "\n")
