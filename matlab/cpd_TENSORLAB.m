% Generate pseudorandom factor matrices U and their associated full tensor T.
size_tens = [7 8 9]; R = 4;
U = cpd_rnd(size_tens,R);
T = cpdgen(U);
% Compute the CPD of the full tensor T.
% Uhat = cpd(T,R);

% Internally, cpd first compresses the tensor
% using a low multilinear rank approximation if it is worthwhile, then chooses a
% method to generate an initialization U0 (e.g., cpd_gevd), after which it executes an algorithm
% to compute the CPD given the initialization (e.g., cpd_nls) and finally decompresses the
% tensor and refines the solution (if compression was applied).

options.Display                     = true;     % Show progress on the command line.
options.Initialization              = @cpd_rnd; % Select pseudorandom initialization.
options.Algorithm                   = @cpd_als; % Select ALS as the main algorithm.
options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
options.AlgorithmOptions.TolFun     = 1e-12;    % Set algorithm stop criteria.
options.AlgorithmOptions.TolX       = 1e-12;

R = rankest(T)

[Uhat,output] = cpd(T,R,options)

semilogy(0:output.Algorithm.iterations,sqrt(2*output.Algorithm.fval));
xlabel('iteration');
ylabel('frob(cpdres(T,U))');
grid on;

relerr = frob(cpdres(T,Uhat))/frob(T)
% computes the subspace angle between the given two sets of factor matrices
sangle = lmlraerr(U,Uhat)
% Uhat{1}...
figure(1); voxel3(T);
figure(2); voxel3(Shat);