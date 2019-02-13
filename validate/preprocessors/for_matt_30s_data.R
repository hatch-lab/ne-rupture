rm(list=ls())
library(ggplot2)
library(cowplot)
library(readxl)
library(splines)
library(plyr)
library(dplyr)
library(stats)

setwd("~/Documents/Hatch Lab/Automated analysis")

true <- read_xlsx("For Matt 30 second data/Analysis/Analysis_061018_30s_Intensity Mean_P2_compiled.xlsx", sheet=2)
names <- true[1,]
true <- true[2:nrow(true),]
names(true) <- names
true$Frame <- as.integer(true$Frame)
true$Frame <- true$Frame*30

validation_data <- read.csv("ne-rupture/validate/validation-data.csv", stringsAsFactors=F)
#validation_data$particle_id <- paste0("matt_30s_", validation_data$particle_id)

validation_data$true_event <- "N"

track_ids <- c(0, 3, 17, 29, 36, 39, 42, 45, 48, 51, 53, 55)
for(track_id in track_ids) {
  rup_col <- paste0(track_id, "R")
  if(rup_col %in% names(true)) {
    true[[rup_col]] <- as.integer(true[[rup_col]])
    rupture_times <- true$Frame[which(true[[rup_col]] == 1)]
    validation_data$true_event[which(validation_data$time %in% rupture_times & validation_data$particle_id == paste0("matt_30s_", track_id))] <- "R"
  }
  
  mit_col <- rup_col <- paste0(track_id, "M")
  if(mit_col %in% names(true)) {
    true[[mit_col]] <- as.integer(true[[mit_col]])
    mitosis_times <- true$Frame[which(true[[mit_col]] == 1)]
    validation_data$true_event[which(validation_data$time %in% mitosis_times & validation_data$particle_id == paste0("matt_30s_", track_id))] <- "M"
  }
  
  death_col <- rup_col <- paste0(track_id, "A")
  if(death_col %in% names(true)) {
    true[[death_col]] <- as.integer(true[[death_col]])
    death_times <- true$Frame[which(true[[death_col]] == 1)]
    validation_data$true_event[which(validation_data$time %in% death_times & validation_data$particle_id == paste0("matt_30s_", track_id))] <- "X"
  }
  
  # Build repairs
  prev_event <- "N"
  curr_event <- "N"
  particle_df <- validation_data[which(validation_data$particle_id == paste0("matt_30s_", track_id)),]
  for(i in 1:nrow(particle_df)) {
    row <- particle_df[i,]
    curr_event <- row$true_event
    
    if(prev_event == "R" && curr_event == "R") {
      if(row$median_derivative > 0 && row$area_derivative < 0) {
        curr_event <- "E"
      }
    
    } else if(prev_event == "E" && curr_event == "R") {
      curr_event <- "E"
    }
    
    validation_data$true_event[which(validation_data$particle_id == paste0("matt_30s_", track_id) & validation_data$time == row$time)] <- curr_event
    prev_event <- curr_event
  }
}

write.csv(validation_data, "ne-rupture/validate/validation-data.csv", row.names=F)


get_groups <- function(df, event_col="true_event") {
  # Walk through rows? There must be a better way
  groups <- data.frame(
    event = character(),
    start = as.numeric(character()),
    end = as.numeric(character()),
    stringsAsFactors=F
  )
  start <- 0
  end <- NA
  old.state <- "NA"
  state <- "NA"
  
  for(i in c(1:nrow(df))) {
    state <- df[i,event_col]
    if(state != old.state) {
      end <- df$time[[i]]
      groups[nrow(groups)+1,] <- c(old.state, start, end)
      start <- df$time[[i]]
    }
    old.state <- state
  }
  groups[nrow(groups)+1,] <- c(old.state, start, max(df$time))
  groups$start <- as.integer(groups$start)
  groups$end <- as.integer(groups$end)
  
  return(groups)
}


track_ids <- c(0, 3, 17, 29, 36, 39, 42, 45, 48, 51, 53, 55)
plots <- list()
for(track_id in track_ids) {
  track_id <- paste0("matt_30s_", track_id)
  groups <- get_groups(validation_data[which(validation_data$particle_id == track_id),])
  groups2 <- get_groups(validation_data[which(validation_data$particle_id == track_id),], "event")
  
  p <- ggplot(validation_data[which(validation_data$particle_id == track_id),], aes(x=time, y=normalized_median)) +
    geom_line() +
    scale_y_continuous(name="") + 
    scale_x_continuous(name="")
  p2 <- ggplot(validation_data[which(validation_data$particle_id == track_id),], aes(x=time, y=normalized_median)) +
    geom_line() +
    scale_y_continuous(name="") + 
    scale_x_continuous(name="")
  for(i in 1:nrow(groups)) {
    event <- groups$event[[i]]
    start <- groups$start[[i]]
    end <- groups$end[[i]]
    if(is.na(event) || event == "NA" || event == "N") {
      next
    }
    
    if(event == "R") {
      color <- "#ffcccc"
    } else if(event == "E") {
      color <- "#ccffcc"
    } else if(event == "X") {
      color <- "#cccccc"
    } else if(event == "M") {
      color <- "#ccccff"
    }
    
    p <- p +
      annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=Inf, color=color, alpha=0.2, fill=color)
  }
  
  for(i in 1:nrow(groups2)) {
    event <- groups2$event[[i]]
    start <- groups2$start[[i]]
    end <- groups2$end[[i]]
    if(is.na(event) || event == "NA" || event == "N") {
      next
    }
    
    if(event == "R") {
      color <- "#ffcccc"
    } else if(event == "E") {
      color <- "#ccffcc"
    } else if(event == "X") {
      color <- "#cccccc"
    } else if(event == "M") {
      color <- "#ccccff"
    }
    
    p2 <- p2 +
      annotate(geom="rect", xmin=start, xmax=end, ymin=-Inf, ymax=Inf, color=color, alpha=0.2, fill=color)
  }
  
  plots[[track_id]] <- plot_grid(p,p2,nrow=2)
}
plot_grid(plotlist=plots)
