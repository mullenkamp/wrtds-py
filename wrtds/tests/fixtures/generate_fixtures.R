#!/usr/bin/env Rscript
# Generate R EGRET reference fixtures for Python comparison tests.
#
# Usage:  Rscript generate_fixtures.R <output_directory>
#
# Requires: EGRET, jsonlite

suppressPackageStartupMessages({
  library(EGRET)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript generate_fixtures.R <output_directory>")
}
out_dir <- args[1]
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

cat("Loading Choptank_eList...\n")
eList <- Choptank_eList

# ---------------------------------------------------------------------------
# 1. Export raw inputs
# ---------------------------------------------------------------------------

cat("Exporting raw inputs...\n")

# Daily: Date, Q
daily_raw <- data.frame(
  Date = as.character(eList$Daily$Date),
  Q = eList$Daily$Q
)
write.csv(daily_raw, file.path(out_dir, "choptank_daily_input.csv"),
          row.names = FALSE)

# Sample: Date, ConcLow, ConcHigh, Uncen
sample_raw <- data.frame(
  Date = as.character(eList$Sample$Date),
  ConcLow = eList$Sample$ConcLow,
  ConcHigh = eList$Sample$ConcHigh,
  Uncen = eList$Sample$Uncen
)
write.csv(sample_raw, file.path(out_dir, "choptank_sample_input.csv"),
          row.names = FALSE)

# ---------------------------------------------------------------------------
# 2. Export INFO metadata
# ---------------------------------------------------------------------------

cat("Exporting INFO metadata...\n")
info_list <- list(
  station_name = eList$INFO$station.nm,
  param_name = eList$INFO$param.nm,
  drainage_area_km2 = eList$INFO$drainSqKm,
  pa_start = as.integer(eList$INFO$paStart),
  pa_long = as.integer(eList$INFO$paLong)
)
write(toJSON(info_list, auto_unbox = TRUE, pretty = TRUE),
      file.path(out_dir, "choptank_info.json"))

# ---------------------------------------------------------------------------
# 3. Re-run modelEstimation with windowY=7
# ---------------------------------------------------------------------------

cat("Running modelEstimation (windowY=7) — this may take several minutes...\n")
eList <- modelEstimation(eList,
                         windowY = 7, windowQ = 2, windowS = 0.5,
                         minNumObs = 100, minNumUncen = 50,
                         edgeAdjust = TRUE, verbose = FALSE)
cat("modelEstimation complete.\n")

# ---------------------------------------------------------------------------
# 4. Export fitted Daily
# ---------------------------------------------------------------------------

cat("Exporting fitted Daily...\n")
daily_fitted <- eList$Daily
# Convert Date to character for clean CSV
daily_fitted$Date <- as.character(daily_fitted$Date)
write.csv(daily_fitted, file.path(out_dir, "choptank_daily_fitted.csv"),
          row.names = FALSE)

# ---------------------------------------------------------------------------
# 5. Export cross-validated Sample
# ---------------------------------------------------------------------------

cat("Exporting cross-validated Sample...\n")
sample_cv <- eList$Sample
sample_cv$Date <- as.character(sample_cv$Date)
write.csv(sample_cv, file.path(out_dir, "choptank_sample_cv.csv"),
          row.names = FALSE)

# ---------------------------------------------------------------------------
# 6. Export surfaces as binary + index as JSON
# ---------------------------------------------------------------------------

cat("Exporting surfaces...\n")
surfaces <- eList$surfaces

# surfaces is a 3D array: (n_logq, n_year, 3)
# Write as raw doubles in Fortran (column-major) order
con <- file(file.path(out_dir, "choptank_surfaces.bin"), "wb")
writeBin(as.double(surfaces), con)
close(con)

# Surface grid parameters
# R stores nVectorLogQ/nVectorYear instead of top values; compute them
n_logq <- dim(surfaces)[1]
n_year <- dim(surfaces)[2]
bottom_logq <- eList$INFO$bottomLogQ
step_logq <- eList$INFO$stepLogQ
bottom_year <- eList$INFO$bottomYear
step_year <- eList$INFO$stepYear

surface_index <- list(
  n_logq = n_logq,
  n_year = n_year,
  n_layer = dim(surfaces)[3],
  bottom_logq = bottom_logq,
  top_logq = bottom_logq + step_logq * (n_logq - 1),
  step_logq = step_logq,
  bottom_year = bottom_year,
  top_year = bottom_year + step_year * (n_year - 1),
  step_year = step_year,
  shape = dim(surfaces)
)
write(toJSON(surface_index, auto_unbox = TRUE, pretty = TRUE),
      file.path(out_dir, "choptank_surface_index.json"))

# ---------------------------------------------------------------------------
# 7. Export setupYears output
# ---------------------------------------------------------------------------

cat("Exporting setupYears...\n")
annual <- setupYears(eList$Daily, paLong = 12, paStart = 10)
annual_out <- data.frame(
  DecYear = annual$DecYear,
  Q = annual$Q,
  Conc = annual$Conc,
  Flux = annual$Flux,
  FNConc = annual$FNConc,
  FNFlux = annual$FNFlux
)
write.csv(annual_out, file.path(out_dir, "choptank_annual.csv"),
          row.names = FALSE)

# ---------------------------------------------------------------------------
# 8. Run WRTDSKalman
# ---------------------------------------------------------------------------

cat("Running WRTDSKalman...\n")
eList_k <- WRTDSKalman(eList, rho = 0.9, niter = 200, seed = 376,
                        verbose = FALSE)

kalman_out <- data.frame(
  Date = as.character(eList_k$Daily$Date),
  GenConc = eList_k$Daily$GenConc,
  GenFlux = eList_k$Daily$GenFlux
)
write.csv(kalman_out, file.path(out_dir, "choptank_daily_kalman.csv"),
          row.names = FALSE)

# ---------------------------------------------------------------------------
# 9. Run runPairs (oldSurface=FALSE, the default — re-estimates narrow surfaces)
# ---------------------------------------------------------------------------

cat("Running runPairs (year1=1985, year2=2010, windowSide=7)...\n")
pairs_result <- runPairs(eList, year1 = 1985, year2 = 2010,
                         windowSide = 7, verbose = FALSE)
write.csv(pairs_result, file.path(out_dir, "choptank_pairs.csv"))

# ---------------------------------------------------------------------------
# 10. Run runGroups (oldSurface=TRUE — reuses existing surface)
# ---------------------------------------------------------------------------

cat("Running runGroups (1985-1996 vs 1997-2010, windowSide=7)...\n")
groups_result <- runGroups(eList, group1firstYear = 1985, group1lastYear = 1996,
                           group2firstYear = 1997, group2lastYear = 2010,
                           windowSide = 7, oldSurface = TRUE, verbose = FALSE)
write.csv(groups_result, file.path(out_dir, "choptank_groups.csv"))

cat("All fixtures generated successfully in:", out_dir, "\n")
