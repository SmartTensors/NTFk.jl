library(dplyr)
library(tidyr)
library(ggplot2)
library(rTensor)
setwd("/Users/alexandrov/Desktop/Proposals/2017_Reserve_Tensor/2018_FY_articles/R_and_Matlab/resultsCPD_X1") 

# Generation of a 3-way tensor rank -1
space_index    <- seq(-1, 1, l = 100)
samples_index  <- seq(0, 90, l = 90)
time_index     <- seq(0, 24, l = 100)

bell_curve  <- dnorm(space_index, mean = 0, sd = 0.5)
#plot(space_index,bell_curve)


case1 <- matrix(rep(bell_curve, 10), 100, 100)
png(filename="case1.png")
image(space_index, time_index, z = case1, xlab = "space", ylab = "time", col = gray((0:32)/32))
contour(space_index, time_index, z = case1, add = TRUE, drawlabels = TRUE)
dev.off()

case2 <- matrix(rep(bell_curve, 10), 100, 100)
case3 <- matrix(rep(bell_curve, 10), 100, 100)
case2[ , 51:100] <- case2[ , 51:100] - 0.1

png(filename="case2.png")
image(space_index, time_index, z = case2, xlab = "space", ylab = "time", col = gray((0:32)/32))
contour(space_index, time_index, z = case1, add = TRUE, drawlabels = TRUE)
dev.off()

case3[ , 51:100] <- case3[ , 51:100] + 0.1

png(filename="case3.png")
image(space_index, time_index, z = case3, xlab = "space", ylab = "time", col = gray((0:32)/32))
contour(space_index, time_index, z = case1, add = TRUE, drawlabels = TRUE)
dev.off()

# X is 3-way: sample-by-space-by-time tensor 

X <- array(NA, dim = c(90, 100, 100))

for(i in 1:30) {
  X[i, , ]    <- case1 + matrix(rnorm(10000, sd = 0.1), 100, 100)
  X[i+30, , ] <- case2 + matrix(rnorm(10000, sd = 0.1), 100, 100)
  X[i+60, , ] <- case3 + matrix(rnorm(10000, sd = 0.1), 100, 100)
}
dim(X)
write.csv(X,"X1.csv",row.names=FALSE) 

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
  ggsave('SpaceTimeX1.png')

  
  
# CPD  of this tensor with rank =1
cpD<- cp(as.tensor(X), num_components = 1)

cpD$conv
cpD$norm_percent

png(filename="Residiums.png")
plot(x<-cpD$all_resids,type ="s", ylab = "Residiums",xlab = "Iterations" )
#points(x, cex = 5, col = "dark blue")
dev.off()
 
write.csv(cpD$U[[1]],"X1_U.csv",row.names=FALSE) 
write.csv(cpD$U[[2]],"X1_V.csv",row.names=FALSE) 
write.csv(cpD$U[[3]],"X1_W.csv",row.names=FALSE) 

png(filename="U[Sample].png")
plot(samples_index,x<-cpD$U[[1]], type = "l", main = "U[Sample]", ylab = "u: The sample component",xlab = "Samples" )
#points(x, cex = 1, col = "dark green")
dev.off()

png(filename="V[Space].png")
plot(space_index, y <- cpD$U[[2]], type = "l", main = "V[Space]", ylab = "v: The space component",xlab = "Space" )
#points(space_index,y, cex = 1, col = "dark red")
dev.off()

png(filename="W[Time].png")
plot(time_index, x <- cpD$U[[3]], type = "l", main = "W[Time]", ylab = "w: The time component",xlab = "Time" )
#points(time_index,x, cex = 1, col = "dark blue")
dev.off()
