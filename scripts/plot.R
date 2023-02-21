require(dplyr)
require(ggplot2)

mytheme = theme(axis.line = element_line(), legend.key=element_rect(fill = NA),
                # text = element_text(size=22),# family = 'PT Sans'),
                # axis.text.x = element_text(size=12),
                # axis.text.y=  element_text(size=12), 
                panel.background = element_rect(fill = "white"))

sl_support_width_plot <- function () {

  resdf <- data.frame(
    lifespan_selsetsize = integer(), 
    low_payoff = double(), 
    sl_support_width = double()
  )

  for (selection_set_size in c(2, 4, 10)) {

    filename <- paste0("csvdata/aggdf_meansoc_B=", selection_set_size, ".csv")

    df <- read.csv(filename)

    new_rows <- df %>%
      filter(low_payoff %in% c(0.1, 0.45)) %>%
      filter(steps_per_round == selection_set_size) %>%
      filter(mean_social_learner < 0.01) %>%
      group_by(steps_per_round, low_payoff) %>%
      slice_min(env_uncertainty)
    
    if (selection_set_size == 2) {
      resdf <- new_rows
    } else {
      resdf <- rbind(resdf, new_rows)
    }
  }

  resdf$low_payoff <- as.factor(resdf$low_payoff)

  g <- ggplot(resdf, aes(x = steps_per_round, y = env_uncertainty, linetype = low_payoff, shape = low_payoff)) +
    geom_point(color="black", size = 4) +
    geom_point(color="white", size = 3, alpha = 0.6) +
    geom_line() + 
    xlab("Selection-set size = effective lifespan") + 
    ylab("SL Extinction Variability") + labs(shape = "Low payoff", linetype="Low payoff") + mytheme

  print(g)
  
  ggsave("~/workspace/Writing/UncMod/Figures/SL_Extinction_Variability.pdf", width = 5, height = 3)

  return (resdf)
}
