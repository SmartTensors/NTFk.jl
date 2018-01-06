import MATLAB
import TensorDecompositions

function ttanalysis(T::Array, crank::Number; seed::Number=1, functionname::String="cp_als", maxiter::Integer=1000, tol::Number=1e-4, printitn::Integer=0, matlabdir::String="/Users/monty/matlab")
	@MATLAB.mput T crank seed
	m = """
	addpath('$matlabdir/TensorToolbox');
	cd('$matlabdir/TensorToolbox');
	rng(seed);
	CP_dec = sptensor(T);
	R = $functionname(CP_dec, crank, struct('maxiters',$maxiter,'tol',$tol,'printitn',$printitn));
	% C = double(R);
	"""
	MATLAB.eval_string(m)
	@MATLAB.mget R
	TT = TensorDecompositions.CANDECOMP((R["u"][1:end]...), R["lambda"])
	# CC = TensorDecompositions.compose(TT)
	# @show maximum(C .- CC)
	return TT
end

function bcuanalysis(T::Array, crank::Number; seed::Number=1, functionname::AbstractString="ncp", maxiter::Integer=1000, tol::Number=1e-4, printitn::Integer=0, matlabdir::String="/Users/monty/matlab")
	@MATLAB.mput T crank seed
	m = """
	addpath('$matlabdir/BCU');
	addpath('$matlabdir/TensorToolbox');
	cd('$matlabdir/TensorToolbox');
	rng(seed);
	CP_dec = sptensor(T);
\	R = $functionname(CP_dec, crank, struct('maxit',$maxiter,'tol',$tol));
	% C = double(R);
	"""
	MATLAB.eval_string(m)
	@MATLAB.mget R
	TT = TensorDecompositions.CANDECOMP((R["u"][1:end]...), R["lambda"])
	# CC = TensorDecompositions.compose(TT)
	# @show maximum(C .- CC)
	return TT
end

function ttanalysis(T::Array, crank::Vector; seed::Number=1, functionname::String="tucker_als", maxiter::Integer=1000, tol::Number=1e-4, printitn::Integer=0, matlabdir::String="/Users/monty/matlab")
	@MATLAB.mput T crank seed
	m = """
	addpath('$matlabdir/TensorToolbox');
	cd('$matlabdir/TensorToolbox');
	rng(seed);
	CP_dec = sptensor(T);
	R = $functionname(CP_dec, crank, struct('maxiters',$maxiter,'tol',$tol,'printitn',$printitn));
	% C = double(R);
	"""
	MATLAB.eval_string(m)
	@MATLAB.mget R
	TT = TensorDecompositions.Tucker((R["u"][1:end]...), R["core"]["data"])
	# CC = TensorDecompositions.compose(TT)
	# @show maximum(C .- CC)
	return TT
end