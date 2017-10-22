library(dplyr)
library(tidyr)
library(ggplot2)
library(rTensor)
setwd("/Users/alexandrov/Desktop/Proposals/2017_Reserve_Tensor/2018_FY_articles/R_and_Matlab/resultsCPD_X3") 
#--- generate data

# bell-shaped spacial component with different means
space_index <- seq(-1, 1, l = 100)
case1 <- matrix(rep(dnorm(space_index, mean = 0, sd = 0.3), 10), 100, 100)
case2 <- matrix(rep(dnorm(space_index, mean = 0.5, sd = 0.3), 10), 100, 100)
case3 <- matrix(rep(dnorm(space_index, mean = -0.5, sd = 0.3), 10), 100, 100)

# sine-shaped temporal component
sine_wave <- sin(seq(-4*pi, 4*pi, l = 100))
sine_mat  <- matrix(rep(sine_wave, each = 100), 100, 100)

case1     <- case1 + 0.3 * sine_mat
case2     <- case2 + 0.6 * sine_mat
case3     <- case3 + 0.9 * sine_mat

# suddent drops in the temporal component
case2[ , 51:100] <- case2[ , 51:100] + 0.1
case3[ , 51:100] <- case3[ , 51:100] - 0.1


png(filename="case1.png")
image(space_index, time_index, z = case1, xlab = "space", ylab = "time", col = gray((0:32)/32))
contour(space_index, time_index, z = case1, add = TRUE, drawlabels = TRUE)
dev.off()


png(filename="case2.png")
image(space_index, time_index, z = case2, xlab = "space", ylab = "time", col = gray((0:32)/32))
contour(space_index, time_index, z = case2, add = TRUE, drawlabels = TRUE)
dev.off()


png(filename="case3.png")
image(space_index, time_index, z = case3, xlab = "space", ylab = "time", col = gray((0:32)/32))
contour(space_index, time_index, z = case3, add = TRUE, drawlabels = TRUE)
dev.off()

# replicate case 1-3 mean data, and augment it with noise,
# in order to obtain a sample for CP analysis;
# organize these data into a 3-way array
X <- array(NA, dim = c(90, 100, 100))
for(i in 1:30) {
  X[i, , ]    <- case1 + matrix(rnorm(10000, sd = 0.1), 100, 100)
  X[i+30, , ] <- case2 + matrix(rnorm(10000, sd = 0.1), 100, 100)
  X[i+60, , ] <- case3 + matrix(rnorm(10000, sd = 0.1), 100, 100)
}

dim(X)
write.csv(X,"X3.csv",row.names=FALSE) 

#definition of a function to visualize case 1, case 2, and case 3 means

case123_to_df <- function(case123, i) {
  as_data_frame(case123) %>%
    mutate(space_index = space_index) %>%
    gather(time_index, Value, -space_index) %>%
    mutate(time_index = as.numeric(gsub("V", "", time_index))) %>%
    mutate(case = i)
}

bind_rows(case123_to_df(case1, "case 1"),
          case123_to_df(case2, "case 2"),
          case123_to_df(case3, "case 3")) %>%
  ggplot(aes(y = space_index, x = time_index, fill = Value)) +
  geom_tile() +
  facet_wrap(~case, nrow = 1) +
  xlab("Time") + ylab("Space") +
  theme(legend.position = "bottom")
  ggsave('SpaceTimeX3.png')
#--- CP decomposition

cp_decomp <- cp(as.tensor(X), num_components = 3, max_iter = 100)

# check convergence status
cp_decomp$conv
# [1] TRUE
# structure of the returned results
str(cp_decomp$U)

for (index in 1:3){
  nameU<-paste0("X3_U",toString(index),".csv")
  write.csv(cp_decomp$U[[1]][ , index], nameU)
  nameV<-paste0("X3_V",toString(index),".csv")
  write.csv(cp_decomp$U[[2]][ , index], nameV)
  nameW<-paste0("X3_W",toString(index),".csv")
  write.csv(cp_decomp$U[[3]][ , index], nameW)
  
}

write.csv(cpD$U[[1]][index],"X3_U.csv",row.names=FALSE) 
write.csv(cpD$U[[2]][index],"X3_V.csv",row.names=FALSE) 
write.csv(cpD$U[[3]][index],"X3_W.csv",row.names=FALSE) 

# percentage of norm explained
cp_decomp$norm_percent

#--- visualize estimated CP components

data_frame(component = c(rep("u[1]", 90), rep("u[2]", 90), rep("u[3]", 90),
                         rep("v[1]", 100), rep("v[2]", 100), rep("v[3]", 100),
                         rep("w[1]", 100), rep("w[2]", 100), rep("w[3]", 100)),
           value = c(cp_decomp$U[[1]][ , 1], cp_decomp$U[[1]][ , 2], cp_decomp$U[[1]][ , 3],
                     cp_decomp$U[[2]][ , 1], cp_decomp$U[[2]][ , 2], cp_decomp$U[[2]][ , 3],
                     cp_decomp$U[[3]][ , 1], cp_decomp$U[[3]][ , 2], cp_decomp$U[[3]][ , 3]),
           index = c(rep(1:90, 3), rep(space_index, 3), rep(1:100, 3))) %>%
  ggplot(aes(index, value)) + geom_line() +
  facet_wrap(~component, scales = "free", nrow = 3,
             labeller = labeller(component = label_parsed)) +
  theme(axis.title = element_blank()) 
  ggsave('U_V_W_X3.png')
