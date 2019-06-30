import MATLAB
import TensorDecompositions
import DocumentFunction

"""
TensorToolbox Candecomp/Parafac (CP) deconstruction

$(DocumentFunction.documentfunction(ttanalysis))
"""
function ttanalysis(T::Array, crank::Number; seed::Number=1, functionname::String="cp_als", maxiter::Integer=DMAXITER, tol::Number=1e-4, printitn::Integer=0, matlabdir::String="/Users/monty/matlab")
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
	if R["lambda"] === nothing
		R["lambda"] = ones(crank)
	end
	if crank == 1
		for i = 1:length(R["u"])
			R["u"][i] = convert(Array{Float64,2}, R["u"][i]')'
		end
		TT = TensorDecompositions.CANDECOMP((R["u"][1:end]...,), vec(collect(R["lambda"])))
	else
		TT = TensorDecompositions.CANDECOMP((R["u"][1:end]...,), R["lambda"])
	end
	# CC = TensorDecompositions.compose(TT)
	# @show maximum(C .- CC)
	return TT
end

"""
TensorToolbox Tucker deconstruction

$(DocumentFunction.documentfunction(ttanalysis))
"""
function ttanalysis(T::Array, crank::Vector; seed::Number=1, functionname::String="tucker_als", maxiter::Integer=DMAXITER, tol::Number=1e-4, printitn::Integer=0, matlabdir::String="/Users/monty/matlab")
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
	TT = TensorDecompositions.Tucker((R["u"][1:end]...,), R["core"]["data"])
	# CC = TensorDecompositions.compose(TT)
	# @show maximum(C .- CC)
	return TT
end

"""
Block-Coordinate Update (BCU) Candecomp/Parafac (CP) deconstruction

$(DocumentFunction.documentfunction(bcuanalysis))
"""
function bcuanalysis(T::Array, crank::Number; seed::Number=1, functionname::AbstractString="ncp", maxiter::Integer=DMAXITER, tol::Number=1e-4, matlabdir::String="/Users/monty/matlab")
	@MATLAB.mput T crank seed
	m = """
	addpath('$matlabdir/BCU');
	addpath('$matlabdir/TensorToolbox');
	cd('$matlabdir/TensorToolbox');
	rng(seed);
	CP_dec = sptensor(T);
	R = $functionname(CP_dec, crank, struct('maxit',$maxiter,'tol',$tol));
	% C = double(R);
	"""
	MATLAB.eval_string(m)
	@MATLAB.mget R
	if R["lambda"] === nothing
		R["lambda"] = ones(crank)
	end
	if crank == 1
		for i = 1:length(R["u"])
			R["u"][i] = convert(Array{Float64,2}, R["u"][i]')'
		end
		TT = TensorDecompositions.CANDECOMP((R["u"][1:end]...,), vec(collect(R["lambda"])))
	else
		TT = TensorDecompositions.CANDECOMP((R["u"][1:end]...,), R["lambda"])
	end
	# CC = TensorDecompositions.compose(TT)
	# @show maximum(C .- CC)
	return TT
end