# Generates cubic splines for the given columns and finds their derivatives

args <- commandArgs(trailingOnly=T)

if(length(args) != 3) {
  stop("3 arguments required")
}

csv_file <- args[1]
columns <- strsplit(args[2], ",")[[1]]
column_stems <- strsplit(args[3], ",")[[1]]

if(length(columns) != length(column_stems)) {
  stop("Need as many columns as column stems")
}

package_list <- c("plyr")
new_packages <- package_list[!(package_list %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages, repos = "http://ftp.ussg.iu.edu/CRAN/")
}

lapply(package_list, require, character.only = TRUE)

###
# Fits cubic splines and calculates derivatives
#
# @param dataframe group DataFrame of each particle
# @param string fit_column The column name to fit
# @param string new_column_stem The prefix of the new columns
# @returns dataframe The modified dataframe
fit_spline <- function(group, fit_column, new_column_stem) {
  x <- group$time
  y <- group[[fit_column]]
  w <- rep.int(1, length(y))
  w[is.na(y)] <- 0
  y[is.na(y)] <- 0
  spline <- smooth.spline(x, y, w=w, spar=0)

  spline_column <- paste0(new_column_stem, '_spline')
  deriv_column <- paste0(new_column_stem, '_derivative')

  group[[spline_column]] <- predict(spline, x)$y
  group[[deriv_column]] <- predict(spline, x, deriv=1)$y

  return(group)
}

fit_splines <- function(group, columns, column_stems) {
  for(idx in 1:length(columns)) {
    group <- fit_spline(group, columns[idx], column_stems[idx])
  }
  return(group)
}

# Figure out column class mapping
MAP <- list(
  "data_set" = "character",
  "particle_id" = "character"
)
MAP_NAMES = names(MAP)
data <- read.csv(csv_file, stringsAsFactors=F, nrows=2)
cols <- names(data)
col_classes <- c()
for(col in cols) {
  if(col %in% MAP_NAMES) {
    col_classes <- c(col_classes, MAP[[col]])
  } else {
    col_classes <- c(col_classes, NA)
  }
}

data <- read.csv(csv_file, stringsAsFactors=F, colClasses=col_classes)

data <- ddply(data, .(data_set, particle_id), fit_splines, columns=columns, column_stems=column_stems)

write.csv(data, csv_file, row.names=FALSE)

