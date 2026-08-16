#!/usr/bin/env Rscript
# DANA cccpcc and voom+limma CCC of -log10(p). Called by stage 05 run.py.
#
# Usage (from repository root):
#   Rscript workflow/05_fidelity/dana_dea.R jobs.csv out.csv
#
# jobs.csv columns:
#   draw, real_csv, gen_csv, coords_csv, reverse_log, tzero, tpoor, twell, failure

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: dana_dea.R jobs.csv out.csv")
}
jobs_path <- args[[1]]
out_path  <- args[[2]]

suppressPackageStartupMessages({
  library(DANA)
  library(DescTools)
  library(limma)
})

listClusters <- function(geneClusters) {
  geneClusters <- as.factor(geneClusters)
  clusters <- lapply(
    levels(geneClusters),
    function(x) names(geneClusters)[geneClusters == x]
  )
  names(clusters) <- levels(geneClusters)
  clusters
}

compute.cc <- function(rawCor, normCor, clusters) {
  shared <- intersect(colnames(rawCor), colnames(normCor))
  if (length(shared) < 2) return(NA_real_)
  rawCor  <- rawCor[shared, shared]
  normCor <- normCor[shared, shared]
  clusters <- clusters[lengths(clusters) > 1]
  raw_v <- c()
  nrm_v <- c()
  for (clust in clusters) {
    g <- clust[clust %in% shared]
    if (length(g) < 2) next
    raw_v <- c(raw_v, rawCor[g, g][upper.tri(rawCor[g, g])])
    nrm_v <- c(nrm_v, normCor[g, g][upper.tri(normCor[g, g])])
  }
  if (length(raw_v) < 2) return(NA_real_)
  DescTools::CCC(raw_v, nrm_v)$rho.c$est
}

load_pair <- function(path) {
  df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  groups <- if ("groups" %in% colnames(df)) as.character(df$groups) else NULL
  keep <- vapply(df, is.numeric, logical(1))
  feat <- as.matrix(df[, keep, drop = FALSE])
  if ("groups" %in% colnames(feat)) {
    feat <- feat[, colnames(feat) != "groups", drop = FALSE]
  }
  list(mat = feat, groups = groups)
}

apply_failure <- function(dat_real, dat_generated, failure) {
  if (failure == "remove") {
    keep <- apply(dat_generated, 2, sd) != 0
    list(
      dat_real[, keep, drop = FALSE],
      dat_generated[, keep, drop = FALSE]
    )
  } else {
    zero_sd <- apply(dat_generated, 2, sd) == 0
    dat_generated[, zero_sd] <- dat_real[, zero_sd]
    list(dat_real, dat_generated)
  }
}

to_counts <- function(mat, reverse_log) {
  if (!reverse_log) return(mat)
  round(pmax(2^mat - 1, 0))
}

ccpos_eval <- function(dat_real, dat_generated, coords, reverse_log, failure,
                       tzero, tpoor, twell) {
  if (is.null(coords) || nrow(coords) == 0) return(NA_real_)
  pair <- apply_failure(dat_real, dat_generated, failure)
  dat_real <- to_counts(pair[[1]], reverse_log)
  dat_generated <- to_counts(pair[[2]], reverse_log)
  colnames(dat_generated) <- colnames(dat_real)
  rownames(dat_generated) <- rownames(dat_real)
  if ((nrow(dat_real) + nrow(dat_generated)) <= 2) return(NA_real_)

  tryCatch({
    clusters <- DANA::defineClusters(
      genes = coords$name,
      chr   = coords$chr,
      pos   = (coords$start + coords$end) / 2
    )
    posControls <- DANA::defineControls(
      raw = t(dat_real), tZero = tzero, tPoor = tpoor, tWell = twell,
      clusters = clusters
    )$posControls
    if (is.null(posControls) || length(posControls) < 2) return(NA_real_)
    invisible(capture.output(
      corPos_real <- DANA::partialCor(t(dat_real[, posControls, drop = FALSE]), scale = TRUE)
    ))
    invisible(capture.output(
      corPos_gen <- DANA::partialCor(t(dat_generated[, posControls, drop = FALSE]), scale = TRUE)
    ))
    compute.cc(corPos_real, corPos_gen, listClusters(as.factor(clusters)))
  }, error = function(e) {
    message("  ccpos_eval: ", conditionMessage(e))
    NA_real_
  })
}

dea_eval <- function(dat_real, dat_generated, groups_real, groups_generated,
                     reverse_log, failure) {
  pair <- apply_failure(dat_real, dat_generated, failure)
  dat_real <- to_counts(pair[[1]], reverse_log)
  dat_generated <- to_counts(pair[[2]], reverse_log)
  genes <- colnames(dat_real)
  if ((nrow(dat_real) + nrow(dat_generated)) <= 2) return(NA_real_)
  if (is.null(groups_real) || is.null(groups_generated)) {
    groups_real <- rep("A", nrow(dat_real))
    groups_real[sample(seq_len(nrow(dat_real)), round(0.5 * nrow(dat_real)))] <- "B"
    groups_generated <- rep("A", nrow(dat_generated))
    groups_generated[sample(seq_len(nrow(dat_generated)), round(0.5 * nrow(dat_generated)))] <- "B"
  }
  tryCatch({
    design_r <- model.matrix(~ factor(groups_real))
    fit_r <- limma::eBayes(limma::lmFit(limma::voom(t(as.matrix(dat_real)), design_r), design_r))
    design_g <- model.matrix(~ factor(groups_generated))
    fit_g <- limma::eBayes(limma::lmFit(limma::voom(t(as.matrix(dat_generated)), design_g), design_g))
    pv_r <- fit_r$p.value[, 2]
    pv_g <- fit_g$p.value[, 2]
    finite_r <- is.finite(log10(pv_r))
    finite_g <- is.finite(log10(pv_g))
    if (any(!finite_r)) pv_r[!finite_r] <- min(pv_r[finite_r], na.rm = TRUE)
    if (any(!finite_g)) pv_g[!finite_g] <- min(pv_g[finite_g], na.rm = TRUE)
    round(DescTools::CCC(-log10(pv_r[genes]), -log10(pv_g[genes]))$rho.c$est, 4)
  }, error = function(e) {
    message("  dea_eval: ", conditionMessage(e))
    NA_real_
  })
}

load_coords <- function(path) {
  if (is.null(path) || !nzchar(path) || !file.exists(path)) return(NULL)
  coords <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  coords$name <- gsub("\\.", "-", coords$name)
  coords
}

jobs <- read.csv(jobs_path, stringsAsFactors = FALSE, check.names = FALSE)
coords_cache <- list()
rows <- list()

for (i in seq_len(nrow(jobs))) {
  job <- jobs[i, ]
  real <- load_pair(job$real_csv)
  gen  <- load_pair(job$gen_csv)
  common <- intersect(colnames(real$mat), colnames(gen$mat))
  dat_r <- real$mat[, common, drop = FALSE]
  dat_g <- gen$mat[, common, drop = FALSE]
  dat_g[!is.finite(dat_g)] <- 0
  dat_g[dat_g < 0] <- 0

  coords <- NULL
  coords_path <- if ("coords_csv" %in% names(job)) as.character(job$coords_csv) else ""
  if (nzchar(coords_path)) {
    if (is.null(coords_cache[[coords_path]])) {
      coords_cache[[coords_path]] <- load_coords(coords_path)
    }
    coords <- coords_cache[[coords_path]]
    if (!is.null(coords)) {
      coords <- coords[coords$name %in% common, , drop = FALSE]
    }
  }

        reverse_log <- tolower(as.character(job$reverse_log)) %in% c("1", "true", "yes", "t")
        failure <- as.character(job$failure)
  tzero <- as.numeric(job$tzero)
  tpoor <- as.numeric(job$tpoor)
  twell <- as.numeric(job$twell)

  cccpcc <- ccpos_eval(dat_r, dat_g, coords, reverse_log, failure, tzero, tpoor, twell)
  ccc_log10 <- dea_eval(dat_r, dat_g, real$groups, gen$groups, reverse_log, failure)

  rows[[length(rows) + 1]] <- data.frame(
    draw = job$draw,
    cccpcc = cccpcc,
    ccc_log10 = ccc_log10,
    stringsAsFactors = FALSE
  )
}

out <- if (length(rows)) do.call(rbind, rows) else {
  data.frame(draw = integer(), cccpcc = numeric(), ccc_log10 = numeric())
}
write.csv(out, out_path, row.names = FALSE)
