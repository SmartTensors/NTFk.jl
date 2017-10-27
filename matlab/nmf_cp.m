addpath(genpath('TensorToolbox'))
rng(2015)
a = rand(20, 1);
b = rand(20, 1);
W = [a b];
H = [.1 1 0 0 .1; 0 0 .1 .5 .2];
Q = [1; 2; 3];
X = W * H;
tsize = [20, 5, 3];
T_orig = zeros(tsize);
T_orig(:,:,1) = X;
T_orig(:,:,2) = X * 2;
T_orig(:,:,3) = X * 3;
cp_nmu(X, 3);