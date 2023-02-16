require(dplyr)
require(ggplot2)

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

  ggplot(resdf, aes(x = steps_per_round, y = env_uncertainty, shape = low_payoff)) +
    geom_point()

  return (resdf)
}
