library(ggplot2)

# PLayers A and B starting with 4 coins, pot with 2
player_A <- c(4)
player_B <- c(4)
players <- c(player_A, player_B)
pot <- c(2)


#function for a single game
playing_one_game <- function(player, pot) {
  i <- sample(1:6)[1]
  if (i == 4 | i == 5 | i == 6) {
    if (player >= 1){
      player <- player - 1
      pot <- pot + 1
    } else {
      return(c(0, player, pot))
    }
  } else if (i == 2) {
    player <- player + pot
    pot <- 0
    #return(1)
  } else if (i == 3) {
    amount <- floor(pot / 2 )
    player <- player + amount
    pot <- pot - amount
    #return(1)
  } else if (i == 1) {
    player <- player
    pot <- pot
    #return(1)
  }
  return(c(1, player, pot))
}


#how many games until a player no longer has coins
cycle_length <- function(players, pot){
  game_list <- 0
  while (result[1] != 0) {
    result <- playing_one_game(players[1], pot)
    players[1] <- result[2]
    pot <- result[3]
    if (result[1] == 0){
      break
    }
    result <- playing_one_game(players[2], pot)
    players[2] <- result[2]
    pot <- result[3]
    game_list <- game_list + 1
  } 
  return(game_list)
}

#checking whether the correct values are being returned
cycle_length(players, pot)


#first run to see whether the loop works. Will be repeated below 
final_results_1000 <- c()
for (i in 1:1000){
  length_result <- cycle_length(players, pot)
  final_results_1000[i] <- length_result
}

df_final_1000 <- data.frame(final_results_1000)
df_final_1000

#####


#empty vectors where final results will be held
final_results_10 <- c()
final_results_100 <- c()
final_results_1000 <- c()
final_results_10000 <- c()
final_results_100000 <- c()


#series of loops (I chose to keep them separate for legibility and to keep R from
#possibly crashing)
#10
for (i in 1:10){
  length_result <- cycle_length(players, pot)
  final_results_10[i] <- length_result
}

#100
for (i in 1:100){
  length_result <- cycle_length(players, pot)
  final_results_100[i] <- length_result
}

#1000
for (i in 1:1000){
  length_result <- cycle_length(players, pot)
  final_results_1000[i] <- length_result
}

#10000
for (i in 1:10000){
  length_result <- cycle_length(players, pot)
  final_results_100000[i] <- length_result
}

#100000
for (i in 1:100000){
  length_result <- cycle_length(players, pot)
  final_results_100000[i] <- length_result
}

#changing to data frame so I can use them with ggplot
df_final_10 <- data.frame(final_results_10)
df_final_100 <- data.frame(final_results_100)
df_final_1000 <- data.frame(final_results_1000)
df_final_10000 <- data.frame(final_results_10000)
df_final_100000 <- data.frame(final_results_100000)


####

#histograms of final result distributions
ggplot(data = df_final_10, aes(x = final_results_10))+
  geom_histogram(color = "black", fill = "cadetblue4")+
  ggtitle("Cycle Length of Game Until a Player Loses (10 Iterations)")+
  xlab("Number of Cycles") + ylab("Count")

ggplot(data = df_final_100, aes(x = final_results_100))+
  geom_histogram(color = "black", fill = "darkseagreen")+
  ggtitle("Cycle Length of Game Until a Player Loses (100 Iterations)")+
  xlab("Number of Cycles") + ylab("Count")


ggplot(data = df_final_1000, aes(x = final_results_1000))+
  geom_histogram(color = "black", fill = "salmon")+
  ggtitle("Cycle Length of Game Until a Player Loses (1,000 Iterations)")+
  xlab("Number of Cycles") + ylab("Count")

ggplot(data = df_final_10000, aes(x = final_results_10000))+
  geom_histogram(color = "black", fill = "pink")+
  ggtitle("Cycle Length of Game Until a Player Loses (10,000 Iterations)")+
  xlab("Number of Cycles") + ylab("Count")

ggplot(data = df_final_100000, aes(x = final_results_100000))+
  geom_histogram(color = "black", fill = "lightblue")+
  ggtitle("Cycle Length of Game Until a Player Loses (100,000 Iterations)")+
  xlab("Number of Cycles") + ylab("Count")



#calculating means, variances, and standard deviations
final_means <- c(mean(final_results_100), 
                 mean(final_results_1000), mean(final_results_10000),
                 mean(final_results_100000))
final_means <- round(final_means, 2)


final_vars <- c(var(final_results_100),
                var(final_results_1000), var(final_results_10000),
                var(final_results_100000))
final_vars <- round(final_vars, 2)


final_sd <- c(sd(final_results_100), 
              sd(final_results_1000), sd(final_results_10000),
              sd(final_results_100000))
final_sd <- round(final_sd, 2)

final_iterations <- c("100", "1,000", "10,000", "100,000")


#combining calculations into dataframes
stats <- data.frame(cbind(final_iterations, 
                          final_means,
                          final_vars,
                          final_sd))


#plots of final calculations
ggplot(data = stats, aes(x = final_iterations, y = final_means))+
  geom_point(color = "red", size = 2)+
  scale_x_discrete(limits = c("100", "1,000", "10,000", "100,000"))+
  ggtitle("Mean of Game Cycles Depending on Number of Iterations")+
  xlab("Iterations") + ylab("Mean")


ggplot(data = stats, aes(x = final_iterations, y = final_sd))+
  geom_point(color = "cadetblue4", size = 2)+
  scale_x_discrete(limits = c("100", "1,000", "10,000", "100,000"))+
  ggtitle("Standard Deviation of Game Cycles Depending on Number of Iterations")+
  xlab("Iterations") + ylab("Standard Deviation")


#checking distribution statistically
library(MASS)
library(vcd)

control <- abs(rnorm(10000))
plot(control)

qqnorm(df_final_10000$final_results_10000, pch = 1)
qqline(df_final_10000$final_results_10000, col = "red", lwd = 2)


#getting lambda values
fit_100 <- fitdistr(final_results_100, "exponential")
fit_1000 <- fitdistr(final_results_1000, "exponential")
fit_10000 <- fitdistr(final_results_10000, "exponential")
fit_100000 <- fitdistr(final_results_100000, "exponential")

fit_100
fit_1000
fit_10000
fit_100000


#to get plot of distributions, mean, and standard deviation
library(fitdistrplus)
descdist(final_results_100, discrete = FALSE)
descdist(final_results_1000, discrete = FALSE)
descdist(final_results_10000, discrete = FALSE)
descdist(final_results_100000, discrete = FALSE)

