import MATLAB

function manalysis(T::Array, crank::Number; seed::Number=1, functionname::String="cp_als")
	@MATLAB.mput T crank seed
	m = """
	addpath('/Users/monty/matlab/TensorToolbox');
	cd('/Users/monty/matlab/TensorToolbox');
	rng(seed);
	CP_dec = sptensor(T);
	R = $functionname(CP_dec, crank);
	% C = double(R);
	"""
	MATLAB.eval_string(m)
	@MATLAB.mget R
	TT = TensorDecompositions.CANDECOMP((R["u"][1:end]...), R["lambda"])
	# CC = TensorDecompositions.compose(TT)
	# @show maximum(C .- CC)
	return TT
end

function manalysis(T::Array, crank::Vector; seed::Number=1, functionname::String="tucker_als")
	@MATLAB.mput T crank seed
	m = """
	addpath('/Users/monty/matlab/TensorToolbox');
	cd('/Users/monty/matlab/TensorToolbox');
	rng(seed);
	CP_dec = sptensor(T);
	R = $functionname(CP_dec, crank);
	% C = double(R);
	"""
	MATLAB.eval_string(m)
	@MATLAB.mget R
	TT = TensorDecompositions.Tucker((R["u"][1:end]...), R["core"]["data"])
	# CC = TensorDecompositions.compose(TT)
	# @show maximum(C .- CC)
	return TT
end