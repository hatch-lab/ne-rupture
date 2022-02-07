# Prints out a graph useful for the bottom of a video

args <- commandArgs(trailingOnly=T)

if(length(args) <= 0) {
  stop("No arguments supplied!", call.=F)
} else if(length(args) != 5) {
  stop("Incorrect number of arguments supplied=!", call.=F)
}

csv_file <- args[1]
movie_plot_output <- args[2]
data_set <- args[3]
particle_id <- args[4]
movie_plot_width <- as.integer(args[5])


##
# Gets contiguous event boundaries
#
# Searches through a time series and finds the start and end
# times for each contiguous block of the same event.
# Returns the event type, start, end, and the associated color.
#
# @param DataFrame df
# @param string event_id_column The event id column to use
# @param string event_column The event column to use
# @returns DataFrame
##
get_event_groups <- function(df, event_id_column="event_id", event_column="event") {
  groups <- data.frame(
    event = character(),
    start = as.numeric(character()),
    end = as.numeric(character()),
    frame_start = as.integer(character()),
    frame_end = as.integer(character()),
    filtered = as.logical(character()),
    color = character(),
    stringsAsFactors=F
  )

  colors <- list(
    "N"="#ffffff",
    "R"="#ffcccc",
    "E"="#ccffcc",
    "X"="#e6e6fa",
    "M"="#ccccff",
    "?"="#eeeeee"
  )

  event_ids <- unique(df[,c(event_id_column, event_column)])
  for(i in 1:nrow(event_ids)) {
    event_id <- event_ids[[event_id_column]][[i]]
    event <- event_ids[[event_column]][[i]]
    event_idx <- which(df[[event_id_column]] == event_id & df[[event_column]] == event)

    start <- min(df$time[event_idx])
    end <- max(df$time[event_idx])
    frame_start <- min(df$frame[event_idx])
    frame_end <- max(df$frame[event_idx])
    event_type <- unique(df[event_idx, c(event_column)])[[1]]

    groups[nrow(groups)+1,] <- c(event_type, start, end, frame_start, frame_end, F, colors[[event_type]])
  }
  
  groups$start <- as.numeric(groups$start)
  groups$end <- as.numeric(groups$end)
  groups$frame_start <- as.integer(groups$frame_start)
  groups$frame_end <- as.integer(groups$frame_end)
  
  return(groups)
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

package_list <- c("ggplot2", "cowplot", "jsonlite", "rlist", "plyr")
new_packages <- package_list[!(package_list %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages, repos = "http://ftp.ussg.iu.edu/CRAN/")
}

quiet_output <- lapply(package_list, library, character.only = TRUE)

theme_set(theme_nothing())

df <- data[which(data$data_set == data_set & data$particle_id == particle_id),]

if(nrow(df) <= 0) {
  stop("No data found matching that data set or cell ID", call.=F)
}

pred_groups <- get_event_groups(df, "event_id", "event")
movie_plot <- ggplot(df, aes(x=frame, y=stationary_mean))

for(i in 1:nrow(pred_groups)) {
  if(!(pred_groups$event[[i]] %in% c("R", "E"))) {
    next
  }
  color <- pred_groups$color[[i]]
  start <- pred_groups$frame_start[[i]]
  end <- pred_groups$frame_end[[i]]+1
  
  movie_plot <- movie_plot + 
    annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=Inf, fill=color, alpha=0.2) 
}

movie_plot <- movie_plot + 
  geom_line(aes(x=frame, y=stationary_cyto_mean), color="#ff4500", alpha=0.9, size=0.1) +
  geom_line(color="#00ffff", alpha=0.8, size=0.1) +
  scale_x_continuous(name="", expand=c(0,0)) +
  scale_y_continuous(name="", expand=c(0,0)) +
  theme_nothing() +
  labs(x = NULL, y = NULL) +
  theme(
    axis.title=element_blank(),
    axis.text=element_blank(),
    axis.ticks=element_blank(),
    axis.line=element_blank(),
    plot.margin=margin(0,0,0,0,"cm"),
    plot.background=element_rect(
      fill="black"
    ),
    panel.border=element_blank()
  )

ggsave(movie_plot_output, movie_plot, width=movie_plot_width, height=50, units="px", device='tiff')


