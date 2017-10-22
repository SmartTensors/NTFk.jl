%% Generate pseudorandom LMLRA (U,S) and associated full tensor T.
size_tens = [17 19 21];
size_core = [3 5 7];
[U,S]     = lmlra_rnd(size_tens,size_core);
T         = lmlragen(U,S);
clear size_core
%% Compute the tensor decomposition
options.Display                 = true;       % Show progress on the command line.
options.Initialization          = @lmlra_rnd; % Select pseudorandom initialization.
options.Algorithm               = @lmlra_nls; % Select NLS as the main algorithm.
options.AlgorithmOptions.TolFun = 1e-12;      % Set stop criteria.
options.AlgorithmOptions.TolX   = 1e-12;

% Plot the estimates of the size of the core tensor (similar to PCA method)
mlrankest(T)
% Estimates the size of the core tensor (as by PCA)
size_core = mlrankest(T) % Optimal core tensor size at L-curve corner.

[Uhat,Shat,output] = lmlra(noisy(T,20),size_core,options);

%% Plot the Norm vs Itterations
semilogy(0:output.Algorithm.iterations,sqrt(2*output.Algorithm.fval));
axis tight;
xlabel('iteration');
ylabel('frob(lmlrares(T,U))');
grid on;
% calculates error in [%]
relerr = frob(lmlrares(T,Uhat,Shat))/frob(T)
% computes the subspace angle between the given two sets of factor matrices
sangle = lmlraerr(U,Uhat)
% Uhat{1}...
figure(1); voxel3(T);
figure(2); voxel3(Shat);
%figure(2); surf3(X);
%figure(3); slice3(X);