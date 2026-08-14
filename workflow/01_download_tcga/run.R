# Stage 01: download TCGA expression cohorts from the GDC API.
#
# For every cohort in config/tcga_cohorts.tsv, queries GDC for the raw
# per-sample quantification files (query spec in config/tcga_download.yml),
# downloads them via the /data endpoint, and merges them into one
# feature-by-sample raw-count matrix per cohort, written to
# data/raw/<modality>/TCGA-<COHORT>.csv. Also writes a per-cohort sample
# manifest (TCGA-<COHORT>.samples.csv) and a run-level download log
# (data/raw/download_log.csv). Cohorts whose output CSV already exists are
# skipped, so an interrupted run can be restarted safely.
#
# Downloads are deterministic, so this stage declares no seed.
#
# Run from the repository root:
#
#   Rscript workflow/01_download_tcga/run.R

library(data.table)
library(httr2)
library(jsonlite)
library(yaml)

# TCGA sample-type codes (positions 14-15 of the barcode), for the sample
# manifest. https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables
SAMPLE_TYPES <- c(
  "01" = "Primary Solid Tumor",
  "02" = "Recurrent Solid Tumor",
  "03" = "Primary Blood Derived Cancer - Peripheral Blood",
  "04" = "Recurrent Blood Derived Cancer - Bone Marrow",
  "05" = "Additional - New Primary",
  "06" = "Metastatic",
  "07" = "Additional Metastatic",
  "08" = "Human Tumor Original Cells",
  "09" = "Primary Blood Derived Cancer - Bone Marrow",
  "10" = "Blood Derived Normal",
  "11" = "Solid Tissue Normal",
  "12" = "Buccal Cell Normal",
  "13" = "EBV Immortalized Normal",
  "14" = "Bone Marrow Normal",
  "20" = "Control Analyte",
  "40" = "Recurrent Blood Derived Cancer - Peripheral Blood",
  "50" = "Cell Lines",
  "60" = "Primary Xenograft Tissue",
  "61" = "Cell Line Derived Xenograft Tissue",
  "99" = "Granulocytes"
)

`%||%` <- function(x, y) if (is.null(x)) y else x

# GDC intermittently answers with HTTP 5xx under load, which httr2 does not
# consider transient by default; retry those with a patient backoff.
gdc_retry <- function(req) {
  req_retry(req,
            max_tries = 6,
            is_transient = function(resp) resp_status(resp) %in% c(429, 500, 502, 503, 504),
            backoff = function(i) min(5 * 2^(i - 1), 120))
}

gdc_filters <- function(cohort, spec) {
  in_filter <- function(field, value) {
    list(op = "in", content = list(field = field, value = I(value)))
  }
  list(
    op = "and",
    content = list(
      in_filter("cases.project.project_id", paste0("TCGA-", cohort)),
      in_filter("data_category", spec$data_category),
      in_filter("data_type", spec$data_type),
      in_filter("analysis.workflow_type", spec$workflow_type)
    )
  )
}

# One row per quantification file: file_id, file_name, aliquot barcode.
query_cohort_files <- function(cfg, cohort, spec) {
  size <- 1000L
  from <- 0L
  hits <- list()
  repeat {
    resp <- request(file.path(cfg$gdc_api, "files")) |>
      req_url_query(
        filters = as.character(toJSON(gdc_filters(cohort, spec), auto_unbox = TRUE)),
        fields = "file_id,file_name,associated_entities.entity_submitter_id",
        size = size,
        from = from,
        format = "JSON"
      ) |>
      gdc_retry() |>
      req_timeout(600) |>
      req_perform()
    body <- resp_body_json(resp)
    hits <- c(hits, body$data$hits)
    from <- from + size
    if (from >= body$data$pagination$total) break
  }
  if (length(hits) == 0) {
    stop("no GDC files found for TCGA-", cohort, " (", spec$data_type, " / ",
         spec$workflow_type, ")")
  }
  files <- rbindlist(lapply(hits, function(h) {
    entities <- vapply(h$associated_entities,
                       function(e) e$entity_submitter_id %||% "", character(1))
    data.frame(
      file_id = h$file_id,
      file_name = h$file_name,
      barcode = grep("^TCGA-", entities, value = TRUE)[1]
    )
  }))
  if (anyNA(files$barcode)) {
    stop("TCGA-", cohort, ": ", sum(is.na(files$barcode)),
         " file(s) without a TCGA aliquot barcode in associated_entities")
  }
  dup <- duplicated(files$barcode)
  if (any(dup)) {
    warning("TCGA-", cohort, ": dropping ", sum(dup),
            " file(s) with duplicate aliquot barcodes: ",
            paste(files$barcode[dup], collapse = ", "))
    files <- files[!dup]
  }
  files[order(files$barcode)]
}

# Download one batch of files into `dest` as <file_id>/<file_name>. The /data
# endpoint returns a tar.gz for multiple IDs but the bare file for a single ID.
download_batch <- function(cfg, files, dest) {
  archive <- file.path(dest, "batch.bin")
  request(file.path(cfg$gdc_api, "data")) |>
    req_method("POST") |>
    req_body_json(list(ids = unname(files$file_id))) |>
    gdc_retry() |>
    req_timeout(3600) |>
    req_perform(path = archive)
  magic <- readBin(archive, "raw", n = 2)
  is_gzip <- length(magic) == 2 && magic[1] == as.raw(0x1f) && magic[2] == as.raw(0x8b)
  if (is_gzip) {
    untar(archive, exdir = dest)
    file.remove(archive)
  } else {
    dir.create(file.path(dest, files$file_id), showWarnings = FALSE)
    file.rename(archive, file.path(dest, files$file_id, files$file_name))
  }
}

# Extract the id/count columns from one quantification file.
read_counts <- function(path, spec) {
  dt <- fread(path, skip = spec$id_column,
              select = c(spec$id_column, spec$count_column))
  setnames(dt, c("id", "count"))
  if (!is.null(spec$keep_pattern)) {
    dt <- dt[grepl(spec$keep_pattern, dt$id)]
  }
  dt
}

download_cohort <- function(cfg, cohort, modality) {
  spec <- cfg$modalities[[modality]]
  if (is.null(spec)) stop("unknown modality: ", modality)

  out_dir <- file.path(cfg$output_dir, modality)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_csv <- file.path(out_dir, sprintf("TCGA-%s.csv", cohort))

  # Cohorts whose matrix is already on disk are not re-downloaded, so an
  # interrupted run can simply be restarted.
  if (file.exists(out_csv)) {
    message(sprintf("[%s/%s] %s exists, skipping", cohort, modality, out_csv))
    features <- nrow(fread(out_csv, select = 1L, showProgress = FALSE))
    samples <- ncol(fread(out_csv, nrows = 0L)) - 1L
    return(data.table(cohort = cohort, modality = modality, files = NA_integer_,
                      features = features, samples = samples, skipped = TRUE,
                      error = NA_character_))
  }

  message(sprintf("[%s/%s] querying GDC", cohort, modality))
  files <- query_cohort_files(cfg, cohort, spec)

  tmp <- tempfile(pattern = "gdc_")
  dir.create(tmp)
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

  feature_ids <- NULL
  columns <- vector("list", nrow(files))
  batches <- split(seq_len(nrow(files)), ceiling(seq_len(nrow(files)) / cfg$batch_size))
  for (b in seq_along(batches)) {
    idx <- batches[[b]]
    message(sprintf("[%s/%s] downloading batch %d/%d (%d files)",
                    cohort, modality, b, length(batches), length(idx)))
    download_batch(cfg, files[idx], tmp)
    for (i in idx) {
      dt <- read_counts(file.path(tmp, files$file_id[i], files$file_name[i]), spec)
      if (is.null(feature_ids)) {
        feature_ids <- dt$id
      } else if (!identical(dt$id, feature_ids)) {
        # Feature order differs between files; realign to the first file.
        dt <- dt[match(feature_ids, dt$id)]
        if (any(is.na(dt$count))) {
          stop("feature ids do not align across files for TCGA-", cohort)
        }
      }
      columns[[i]] <- as.integer(dt$count)
    }
    unlink(file.path(tmp, files$file_id[idx]), recursive = TRUE)
    file.remove(file.path(tmp, "MANIFEST.txt"))
  }

  mat <- matrix(unlist(columns), ncol = length(columns),
                dimnames = list(feature_ids, files$barcode))

  message(sprintf("[%s/%s] writing %s (%d features x %d samples)",
                  cohort, modality, out_csv, nrow(mat), ncol(mat)))
  # Write-then-rename so an interrupted run never leaves a truncated matrix
  # that the skip-above logic would mistake for complete.
  tmp_csv <- paste0(out_csv, ".tmp")
  write.csv(as.data.frame(mat), tmp_csv)
  file.rename(tmp_csv, out_csv)

  manifest <- data.table(
    barcode = files$barcode,
    sample_type = unname(SAMPLE_TYPES[substr(files$barcode, 14, 15)]),
    file_id = files$file_id,
    file_name = files$file_name
  )
  fwrite(manifest, file.path(out_dir, sprintf("TCGA-%s.samples.csv", cohort)))

  data.table(cohort = cohort, modality = modality, files = nrow(files),
             features = nrow(mat), samples = ncol(mat), skipped = FALSE,
             error = NA_character_)
}

gdc_data_release <- function(cfg) {
  tryCatch(
    resp_body_json(req_perform(request(file.path(cfg$gdc_api, "status"))))$data_release %||% NA_character_,
    error = function(e) NA_character_
  )
}

main <- function() {
  if (!file.exists("config/tcga_cohorts.tsv")) {
    stop("config/tcga_cohorts.tsv not found; run from the repository root")
  }
  cfg <- read_yaml("config/tcga_download.yml")
  cohorts <- read.delim("config/tcga_cohorts.tsv", stringsAsFactors = FALSE)
  # A failing cohort is recorded in the log and does not stop the rest;
  # rerun the script to retry it.
  log <- rbindlist(lapply(seq_len(nrow(cohorts)), function(i) {
    tryCatch(
      download_cohort(cfg, cohorts$cohort[i], cohorts$modality[i]),
      error = function(e) {
        message(sprintf("[%s/%s] FAILED: %s", cohorts$cohort[i],
                        cohorts$modality[i], conditionMessage(e)))
        data.table(cohort = cohorts$cohort[i], modality = cohorts$modality[i],
                   files = NA_integer_, features = NA_integer_,
                   samples = NA_integer_, skipped = FALSE,
                   error = conditionMessage(e))
      }
    )
  }))
  log[, `:=`(gdc_data_release = gdc_data_release(cfg),
             downloaded_at = format(Sys.time(), tz = "UTC", usetz = TRUE))]
  fwrite(log, file.path(cfg$output_dir, "download_log.csv"))
  n_failed <- sum(!is.na(log$error))
  if (n_failed > 0) {
    stop(n_failed, " cohort(s) failed: ",
         paste(log$cohort[!is.na(log$error)], collapse = ", "),
         "; rerun to retry")
  }
  message("done: ", nrow(log), " cohorts written to ", cfg$output_dir)
}

if (sys.nframe() == 0L) main()
