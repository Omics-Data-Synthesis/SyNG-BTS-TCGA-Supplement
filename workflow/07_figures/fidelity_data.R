# Shared reader for the Stage 05 signed-source-ARI table.
read_fidelity <- function(root) {
  dat <- read.csv(file.path(root, "results", "fidelity.csv"), stringsAsFactors = FALSE)
  keys <- c("dataset_id", "modality", "norm", "model", "off_aug", "AE_head_num")
  required <- c(keys, "draw", "draw_seed", "source_ari", "seed",
                "n_real", "n_generated", "n_features", "cccpcc", "ccc_log10_pvalue",
                "ccc_log2_FC", "ccc_meandiff", "mad_mean", "mad_sd", "mad_cv",
                "medae_mean", "medae_sd", "medae_cv")
  if (!setequal(required, names(dat)))
    stop("Incompatible fidelity schema: recompute signed source ARI.")
  if (!nrow(dat) || !is.numeric(dat$source_ari) ||
      any(!is.finite(dat$source_ari) | dat$source_ari < -.5 | dat$source_ari > 1))
    stop("Missing or invalid signed source ARI.")
  cfg <- read.delim(file.path(root, "config", "fidelity.tsv"))
  n_draws <- as.integer(cfg$value[cfg$param == "n_draws"])
  seed_base <- as.integer(cfg$value[cfg$param == "draw_seed_base"])
  dat$AE_head_num[is.na(dat$AE_head_num)] <- 0
  counts <- dplyr::count(dat, dplyr::across(dplyr::all_of(keys)))
  if (any(counts$n != n_draws) || anyDuplicated(dat[c(keys, "draw")]) ||
      anyNA(dat[c(keys, "draw", "draw_seed")]) ||
      any(!dat$draw %in% seq_len(n_draws)) ||
      any(dat$draw_seed != seed_base + dat$draw))
    stop("Incomplete, duplicate or incompatible fidelity draws.")
  dat
}

ari_label <- function(x) {
  # Show small signed values explicitly; never format a negative value as -0.00.
  ifelse(x == 0, "0.00", ifelse(x < 0 & abs(x) < .005,
         formatC(x, format = "g", digits = 2), sprintf("%.2f", x)))
}
