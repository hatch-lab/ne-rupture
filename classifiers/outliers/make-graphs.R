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

data <- read.csv(csv_file, stringsAsFactors=F, colClasses=col_classes)

have_true_events <- T
if(!("true_event" %in% names(data))) {
  data$true_event <- "N"
  have_true_events <- F
}

package_list <- c("ggplot2", "cowplot", "jsonlite", "rlist", "plyr")
new_packages <- package_list[!(package_list %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages)
}

lapply(package_list, require, character.only = TRUE)

raw_conf <- readLines(conf_path, warn=F)
classifier_conf <- fromJSON(raw_conf)

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

    groups[nrow(groups)+1,] <- c(event_type, start, end, frame_start, frame_end, colors[[event_type]])
  }
  
  groups$start <- as.numeric(groups$start)
  groups$end <- as.numeric(groups$end)
  groups$frame_start <- as.integer(groups$frame_start)
  groups$frame_end <- as.integer(groups$frame_end)
  
  return(groups)
}

##
# Returns function to format seconds
#
# This is actually a wrapper around the actual function,
# which is necessary for ggplot2
##
format_time_labels <- function() {
  return(function(x) {
    # x is in seconds
    hours <- x %/% 3600
    minutes <- (x-(hours*3600)) %/% 60
    seconds <- (x-(hours*3600)) %% 60
    
    return(paste0(sprintf("%02dh", hours), sprintf("%02d′", minutes), sprintf("%02d″", seconds)))
  })
}

##
# Returns function to format frames
#
# This is actually a wrapper around the actual function,
# which is necessary for ggplot2
##
format_frame_labels <- function() {
  return(function(x) {
    return(floor(x))
  })
}

# Generate a list of plots for each particle
interesting_plots <- list()
plots <- list()
data_sets <- unique(data$data_set)
for(m in 1:length(data_sets)) {
  data_set <- data_sets[[m]]
  pids <- unique(data$particle_id[which(data$data_set == data_set)])
  frame_rate <- unique(data$frame_rate[which(data$data_set == data_set)])[[1]]
  for(n in 1:length(pids)) {
    is_interesting <- F
    pid <- pids[[n]]
    df <- data[which(data$particle_id == pid & data$data_set == data_set),]
    
    base_plot <- ggplot(df, aes(x=time, y=normalized_median))
    
    # Annotate the plot with predicted and true events
    pred_groups <- get_event_groups(df, "event_id", "event")
    if(have_true_events) {
      true_groups <- get_event_groups(df, "true_event_id", "true_event") 
    } else {
      true_groups <- list()
    }
    for(i in 1:nrow(pred_groups)) {
      color <- pred_groups$color[[i]]
      start <- pred_groups$start[[i]]
      end <- pred_groups$end[[i]]
      event <- pred_groups$event[[i]]
      
      if(event != "N") {
        is_interesting <- T
      }
      
      if(have_true_events) {
        base_plot <- base_plot + 
          annotate(geom="rect", xmin=start, xmax=end, ymin=0, ymax=Inf, color=color, fill=color) 
      } else {
        base_plot <- base_plot + 
          annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=Inf, color=color, fill=color)
      }
    }
    if(have_true_events) {
      for(i in 1:nrow(true_groups)) {
        color <- true_groups$color[[i]]
        start <- true_groups$start[[i]]
        end <- true_groups$end[[i]]
        event <- true_groups$event[[i]]
        
        if(event != "N") {
          is_interesting <- T
        }
        
        base_plot <- base_plot + 
          annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=0, color=color, fill=color)
      }
      
      base_plot <- base_plot + 
        annotate(geom="segment", x=-Inf, xend=Inf, y=0, yend=0, color="#cccccc") 
    }
    
    # Generate plots
    median_cutoff <- classifier_conf[['median_cutoff']]
    baseline_median <- mean(df$stationary_median[which(df$event == "N")], na.rm=T)
    median_annotation <- quantile(df$stationary_median, 0.25, na.rm=T)[["25%"]]-median_cutoff*IQR(df$stationary_median, na.rm=T)
    median_limit <- max(c(
      abs(min(c(df$stationary_median*1.5, median_annotation*1.5), na.rm=T)), 
      abs(max(c(df$stationary_median*1.5, median_annotation*1.5), na.rm=T))
    ), na.rm=T)
    
    median_plot <- base_plot +
      geom_line(aes(x=time, y=stationary_median)) +
      ggtitle(paste0(data_set, ":", pid)) +
      scale_y_continuous(name="Median", limits=c(-median_limit, median_limit)) +
      scale_x_continuous(name="", labels=format_time_labels(), sec.axis = sec_axis(~./frame_rate, name="Frame", labels=format_frame_labels())) +
      annotate(geom="segment", x=-Inf, xend=Inf, y=median_annotation, yend=median_annotation, color="red", alpha=0.7) +
      annotate(geom="segment", x=-Inf, xend=Inf, y=baseline_median, yend=baseline_median, linetype="dashed", alpha=0.7)
    if(have_true_events) {
      median_plot <- median_plot +
        annotate(geom="text", x=0.1, y=0.5, label="Predicted", alpha=0.8, hjust=0) +
        annotate(geom="text", x=0.1, y=-0.2, label="True", alpha=0.8, hjust=0)
    }

    area_cutoff <- classifier_conf[['area_cutoff']]
    baseline_area <- mean(df$stationary_area[which(df$event == "N")], na.rm=T)
    area_annotation <- quantile(df$stationary_area, 0.75, na.rm=T)[["75%"]]+area_cutoff*IQR(df$stationary_area, na.rm=T)[[1]]
    area_limit <- max(c(
      abs(min(c(df$stationary_area*1.5, area_annotation*1.5), na.rm=T)), 
      abs(max(c(df$stationary_area*1.5, area_annotation*1.5), na.rm=T))
    ), na.rm=T)
    
    area_plot <- base_plot +
      geom_line(aes(x=time, y=stationary_area)) +
      annotate(geom="segment", x=-Inf, xend=Inf, y=area_annotation, yend=area_annotation, color="red", alpha=0.7) +
      annotate(geom="segment", x=-Inf, xend=Inf, y=baseline_area, yend=baseline_area, linetype="dashed", alpha=0.7) +
      scale_y_continuous(name="Area", limits=c(-area_limit, area_limit)) +
      scale_x_continuous(name="", labels=format_time_labels(), sec.axis = sec_axis(~./frame_rate, name="", labels=format_frame_labels()))
    
    plots[[(length(plots)+1)]] <- plot_grid(median_plot, area_plot, nrow=2, align="v", rel_heights=c(1.4,1.8))

    # Plots to put under movies
    movie_plot <- ggplot(df, aes(x=frame, y=stationary_median))
    for(i in 1:nrow(pred_groups)) {
      if(!(pred_groups$event[[i]] %in% c("R", "E"))) {
        next
      }
      color <- "#ffffff"
      start <- pred_groups$frame_start[[i]]
      end <- pred_groups$frame_end[[i]]+1
      
      if(have_true_events) {
        movie_plot <- movie_plot + 
          annotate(geom="rect", xmin=start, xmax=end, ymin=0, ymax=Inf, fill=color, alpha=0.2)
      } else {
        movie_plot <- movie_plot + 
          annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=Inf, fill=color, alpha=0.2) 
      }
    }
    if(have_true_events) {
      for(i in 1:nrow(true_groups)) {
        if(!(true_groups$event[[i]] %in% c("R", "E"))) {
          next
        }
        color <- "#ffffff"
        start <- true_groups$frame_start[[i]]
        end <- true_groups$frame_end[[i]]+1
        
        movie_plot <- movie_plot + 
          annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=0, fill=color, alpha=0.2)
      } 
    }
    movie_plot <- movie_plot + 
      geom_line(aes(x=frame, y=stationary_area), color="#ff4500", alpha=0.9, size=0.1) +
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
    
    if(is_interesting) {
      interesting_plots[[length(interesting_plots)+1]] <- base_plot +
        geom_line(aes(x=time, y=stationary_median)) +
        ggtitle(paste0(strtrim(data_set, 5), ":", pid)) +
        scale_y_continuous(name="", limits=c(-1,1), labels=c()) +
        scale_x_continuous(name="", labels=c()) +
        theme(axis.title=element_blank(),
              axis.text=element_blank(),
              axis.ticks=element_blank(),
              axis.line=element_blank())
    }
  } 
}

if(length(interesting_plots) > 0) {
  plots <- list.prepend(plots, list(plot_grid(plotlist=interesting_plots, align="hv")))
}

# Print out
cairo_pdf(output, width=10, height=8.25, onefile=T)
for(i in 1:length(plots)) {
  print(plots[[i]])
}
dev.off()


