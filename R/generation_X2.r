library(parallel)
detectCores()
library(rTensor)
setwd("/home/boian/Desktop/R_tensors/resultsCPD_X2") 

# Generation of a 3-way tensor with rank-3

samples_index  <- seq(0, 90,     l = 90)
space_index    <- seq(-1.5, 1.5, l = 100)
time_index     <- seq(0, 100,    l = 100)

gama <- 0.5

bell_curve1  <- dnorm(space_index, mean =  0, sd = gama)
bell_curve2  <- dnorm(space_index, mean = -1, sd = gama)
bell_curve3  <- dnorm(space_index, mean =  1, sd = gama)


alpha <- 4.0
beta  <- 15.0
delta <- 0.1
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

case1 <- matrix(rep(bell_curve1, 10), 100, 100)
case1 <- beta*sin(pi*time_index/alpha)*case1
image(time_index, space_index,case1, main = "P1", xla = "time", yla = "space")
 
case2 <- matrix(rep(bell_curve2, 10), 100, 100)
case2[ , 51:100] <- case2[ , 51:100] + delta
case2 <- beta*sin(pi*time_index/alpha)*case2
image(time_index, space_index,case2, main = "P2", xla = "time", yla = "space")

case3 <- matrix(rep(bell_curve3, 10), 100, 100)
case3[ , 51:100] <- case3[ , 51:100] - delta
case3 <- beta*sin(pi*time_index/alpha)*case3
image(time_index, space_index,case3, main = "P3", xla = "time", yla = "space")




#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# X is 3-way: sample-by-space-by-time tensor 

X <- array(NA, dim = c(90, 100, 100))

for(i in 1:30) {
  X[i, , ]    <- case1 + matrix(rnorm(10000, sd = 0.1), 100, 100)
  X[i+30, , ] <- case2 + matrix(rnorm(10000, sd = 0.1), 100, 100)
  X[i+60, , ] <- case3 + matrix(rnorm(10000, sd = 0.1), 100, 100)
}
dim(X)

write.csv(X,"X2.csv") 
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# CPD  of this tensor with rank =1
cpD<- cp(as.tensor(X), num_components = 3, max_iter = 100, tol = 1e-3)

cpD$conv
cpD$norm_percent
plot(x<-cpD$all_resids,type ="s", ylab = "Residiums",xlab = "Iterations" )
points(x, cex = 2, col = "dark blue")

write.csv(cpD$U[[1]],"X2_U.csv") 
write.csv(cpD$U[[2]],"X3_V.csv") 
write.csv(cpD$U[[3]],"X4_W.csv") 

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

plot(samples_index,x<-cpD$U[[1]] [,1] , type = "s", main = "U[1, Sample]", ylab = "u: The sample component",xlab = "Samples" )
points(x, cex = 1, col = "dark green")

plot(samples_index,x<-cpD$U[[1]] [,2] , type = "s", main = "U[2, Sample]", ylab = "u: The sample component",xlab = "Samples" )
points(x, cex = 1, col = "dark green")

plot(samples_index,x<-cpD$U[[1]] [,3] , type = "s", main = "U[3, Sample]", ylab = "u: The sample component",xlab = "Samples" )
points(x, cex = 1, col = "dark green")

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

plot(space_index, y <- cpD$U[[2]][,1] , type = "s", main = "V[1, Space]", ylab = "v: The space component",xlab = "Space" )
points(space_index,y, cex = 1, col = "dark red")

plot(space_index, y <- cpD$U[[2]][,2] , type = "s", main = "V[2, Space]", ylab = "v: The space component",xlab = "Space" )
points(space_index,y, cex = 1, col = "dark red")

plot(space_index, y <- cpD$U[[2]][,3] , type = "s", main = "V[3, Space]", ylab = "v: The space component",xlab = "Space" )
points(space_index,y, cex = 1, col = "dark red")

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

plot(time_index, x <- cpD$U[[3]][,1] , type = "l", main = "W[1, Time]", ylab = "w: The time component",xlab = "Time" )
points(time_index,x, cex = 1, col = "dark blue")

plot(time_index, x <- cpD$U[[3]][,2] , type = "l", main = "W[2, Time]", ylab = "w: The time component",xlab = "Time" )
points(time_index,x, cex = 1, col = "dark blue")

plot(time_index, x <- cpD$U[[3]][,3] , type = "l", main = "W[3, Time]", ylab = "w: The time component",xlab = "Time" )
points(time_index,x, cex = 1, col = "dark blue")

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx