# NFELO FILM MARGIN
nfelo film margins are margins derived from a linear regression of various PFF grades and the final marign of victory for each team
* film_margin is a descriptive margin that uses film grades from a single game to estimate the actual result of that game. The RSQ of this model is about 0.66
* film_margin_predictive is a predictive model that uses film grades from a single game to predict the average margin of victory for all other games for a team in the season. The RSQ of this model is about 0.15
* film_margin_old_model is the previous version of film_margin, built only from the overall grade. The RSQ of this model is about 0.56