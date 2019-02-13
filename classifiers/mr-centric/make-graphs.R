# Prints out a list of graphs comparing true and predicted values
# with the truth underneath and the prediction on top

args <- commandArgs(trailingOnly=T)

if(length(args) <= 0) {
  stop("No arguments supplied!", call.=F)
} else if(length(args) != 5) {
  stop("Incorrect number of arguments supplied=!", call.=F)
}

csv_file <- args[1]
output <- args[2]
movie_plot_width <- as.integer(args[3])
movie_plot_output <- args[4]
conf_path <- args[5]

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

COLORS <- list(
  "N"="#ffffff",
  "R"="#ffcccc",
  "E"="#ccffcc",
  "X"="#cccccc",
  "MR"="#cccccc",
  "M"="#ccccff"
)

data <- read.csv(csv_file, stringsAsFactors=F, colClasses=col_classes)

have_true_events <- T
if(!("true_event" %in% names(data))) {
  data$true_event <- "N"
  data$true_event_id <- -1
  have_true_events <- F
}

library(ggplot2)
library(cowplot)
library(jsonlite)

raw_conf <- readLines(conf_path, warn=F)
classifier_conf <- fromJSON(raw_conf)

# Generate a list of plots for each particle
plots <- list()
data_sets <- unique(data$data_set)
for(m in 1:length(data_sets)) {
  data_set <- data_sets[[m]]
  pids <- unique(data$particle_id[which(data$data_set == data_set)])
  for(n in 1:length(pids)) {
    pid <- pids[[n]]
    df <- data[which(data$particle_id == pid & data$data_set == data_set),]
    
    base_plot <- ggplot(df, aes(x=time, y=normalized_median))
    # Derivative plots are usually very, very small, so our event annotations need to be different
    deriv_plot <- ggplot(df, aes(x=time, y=normalized_median))
    # Plots to put under movies
    movie_plot <- ggplot(df, aes(x=frame, y=normalized_median))
    
    # Annotate the plot with predicted and true events
    event_ids <- unique(df$event_id[which(df$event_id != -1)])
    true_event_ids <- unique(df$true_event_id[which(df$true_event_id != -1)])

    for(i in event_ids) {
      events <- df$event[which(df$event_id == i)]

      for(event in events) {
        color <- COLORS[[event]]
        start <- min(df$time[which(df$event_id == i & df$event == event)])
        end <- max(df$time[which(df$event_id == i & df$event == event)]) + diff(df$time)[[1]]
        if(have_true_events) {
          base_plot <- base_plot + 
            annotate(geom="rect", xmin=start, xmax=end, ymin=0.5, ymax=Inf, color="black", fill=color) 
        } else {
          base_plot <- base_plot + 
            annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=Inf, color="black", fill=color)
        }
        deriv_plot <- deriv_plot + 
          annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=Inf, color="black", fill=color)

        if(!(event %in% c("R", "E"))) {
          next
        }
        
        # Movie plots use frames, not time
        start <- min(df$frame[which(df$event_id == i & df$event == event)])
        end <- max(df$frame[which(df$event_id == i & df$event == event)]) + diff(df$time)[[1]]
        if(have_true_events) {
          movie_plot <- movie_plot + 
            annotate(geom="rect", xmin=start, xmax=end, ymin=0.5, ymax=Inf, fill="white", alpha=0.2)
        } else {
          movie_plot <- movie_plot + 
            annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=Inf, fill="white", alpha=0.2)
        }
      }
    }
    if(have_true_events) {
      for(i in true_event_ids) {
        events <- df$true_event[which(df$true_event_id == i)]

        for(event in events) {
          color <- COLORS[[event]]
          start <- min(df$time[which(df$event_id == i & df$true_event == event)])
          end <- max(df$time[which(df$event_id == i & df$true_event == event)]) + diff(df$time)[[1]]
          base_plot <- base_plot + 
            annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=0.5, color="black", fill=color)
          movie_plot <- movie_plot + 
            annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=0.5, fill="white", alpha=0.2)
        }
      }
    }
    
    if(have_true_events) {
      base_plot <- base_plot + 
        annotate(geom="segment", x=-Inf, xend=Inf, y=0.5, yend=0.5, color="#cccccc") 
    }
    
    # Generate median, area, summ, and ellipticity plots
    baseline_median <- mean(df$normalized_median[which(df$event == "N")])
    median_plot <- base_plot +
      geom_line(aes(x=time, y=normalized_median)) +
      ggtitle(paste0(data_set, ":", pid)) +
      scale_y_continuous(name="Median", limits=c(0,1)) +
      scale_x_continuous(name="") +
      annotate(geom="segment", x=-Inf, xend=Inf, y=baseline_median, yend=baseline_median, linetype="dashed", alpha=0.7)
    if(have_true_events) {
      median_plot <- median_plot +
        annotate(geom="text", x=0.1, y=0.9, label="Predicted", alpha=0.8, hjust=0) +
        annotate(geom="text", x=0.1, y=0.1, label="True", alpha=0.8, hjust=0)
    }
    
    area_plot <- base_plot +
      geom_line(aes(x=time, y=scaled_area)) +
      scale_y_continuous(name="Area", limits=c(0,1)) +
      scale_x_continuous(name="")
    
    median_cutoff <- classifier_conf$median_derivative
    area_cutoff <- classifier_conf$area_derivative
    rupture_deriv_plot <- deriv_plot + 
      geom_line(aes(x=time, y=median_derivative), color="black") +
      geom_line(aes(x=time, y=area_derivative), color="#b42f32") +
      scale_y_continuous(name="Rupture cutoffs", limits=c(min(c(median_cutoff, df$median_derivative, df$area_derivative))*1.05, max(c(area_cutoff, df$median_derivative, df$area_derivative))*1.05)) +
      scale_x_continuous(name="Time (s)") +
      annotate(geom="segment", x=-Inf, xend=Inf, y=median_cutoff, yend=median_cutoff, color="black", alpha=0.7, linetype="dashed") +
      annotate(geom="segment", x=-Inf, xend=Inf, y=area_cutoff, yend=area_cutoff, color="#b42f32", alpha=0.7, linetype="dashed") 
    
    plots[[(length(plots)+1)]] <- plot_grid(median_plot, area_plot, rupture_deriv_plot, nrow=3, align="v", rel_heights=c(1.4,1,1,2))

    movie_plot <- movie_plot + 
      geom_line(aes(x=frame, y=scaled_area), color="#ff4500", alpha=0.9, size=0.1) +
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
    ggsave(paste0(movie_plot_output, "/", data_set, "_", pid, ".tiff"), movie_plot, width=movie_plot_width/300, height=50/300, units="in") 
  } 
}

# Print out
pdf(output, width=10, height=8.25, onefile=T)
for(i in 1:length(plots)) {
  print(plots[[i]])
}
dev.off()


