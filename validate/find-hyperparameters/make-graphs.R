# Prints out AUC curves following a find-hyperparameters search

args <- commandArgs(trailingOnly=T)

if(length(args) <= 0) {
  stop("No arguments supplied!", call.=F)
} else if(length(args) != 2) {
  stop("Incorrect number of arguments supplied=!", call.=F)
}

csv_file <- args[1]
output <- args[2]

package_list <- c("ggplot2", "cowplot", "ggforce")
new_packages <- package_list[!(package_list %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages)
}

lapply(package_list, require, character.only = TRUE)

auc <- read.csv(csv_file, stringsAsFactors=F)
auc$x <- 1-auc$specificity
auc <- auc[order(auc$sensitivity),]

idx <- which(auc$sensitivity > 0.8 & auc$data_set == "All" & auc$event == "Rupture")
best_df <- auc[idx,]
best_df <- best_df[order(best_df$x),]
best_df <- best_df[1,]

best_text <- paste0("sensitivity: ", best_df$sensitivity[[1]], "\n", "1-specificity: ", best_df$x[[1]], "\n")
for(param in setdiff(names(best_df),c("data_set", "event", "sensitivity", "specificity", "x"))) {
  best_text <- paste0(best_text, param, ": ", round(best_df[[param]][[1]], 4), "\n")
}

base_plot <- ggplot(auc[which(auc$event == "Rupture" & auc$data_set == "All"),], aes(x=x, y=sensitivity)) +
  geom_point(alpha=0.3) +
  geom_point(data=best_df, color="red") +
  scale_x_continuous(name="1-specificity") +
  scale_y_continuous(name="sensitivity") +
  annotate("segment", x=0, xend=1, y=0, yend=1, color="gray") +
  theme(legend.position = "none") +
  annotate("text", x=1, y=0, label=best_text, hjust=1, vjust=0)

# Print out
cairo_pdf(output, width=10, height=8.25, onefile=T)
print(base_plot)
dev.off()
