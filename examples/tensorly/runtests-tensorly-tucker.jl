import NTFk
import PyCall
import TensorDecompositions

@PyCall.pyimport tensorly as tl
@PyCall.pyimport tensorly.decomposition as td

srand(1)
csize = (10,10,10)
tsize = (1000, 1000, 1000)
tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(tucker_orig);

info("numpy")
tl.set_backend("numpy")
@time T_orig_tl = tl.backend[:tensor](T_orig);
@time td.non_negative_tucker(T_orig_tl, rank=csize, n_iter_max=2);

info("mxnet")
tl.set_backend("mxnet")
@time T_orig_tl = tl.backend[:tensor](T_orig);
@time td.non_negative_tucker(T_orig_tl, rank=csize, n_iter_max=2);

info("pytorch")
tl.set_backend("pytorch")
@time T_orig_tl = tl.backend[:tensor](T_orig);
@time td.non_negative_tucker(T_orig_tl, rank=csize, n_iter_max=2);

info("tensorflow")
tl.set_backend("tensorflow");
@time T_orig_tl = tl.backend[:tensor](T_orig);
@time td.non_negative_tucker(T_orig_tl, rank=csize, n_iter_max=2);

info("numpy")
tl.set_backend("numpy")
@time td.non_negative_tucker(tl.backend[:tensor](T_orig), rank=csize, n_iter_max=2);

info("mxnet")
tl.set_backend("mxnet")
@time td.non_negative_tucker(tl.backend[:tensor](T_orig), rank=csize, n_iter_max=2);

info("pytorch")
tl.set_backend("pytorch")
@time td.non_negative_tucker(tl.backend[:tensor](T_orig), rank=csize, n_iter_max=2);

info("tensorflow")
tl.set_backend("tensorflow");
@time td.non_negative_tucker(tl.backend[:tensor](T_orig), rank=csize, n_iter_max=2);

NTFk.tlanalysis(T_orig, [csize...]; maxiter=2)

NTFk.analysis(T_orig, [csize], 1; progressbar=false, tol=1e-4, max_iter=2)
