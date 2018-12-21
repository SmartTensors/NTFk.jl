import NTFk
import TensorDecompositions

Random.seed!(1)
csize = (2, 3, 4)
tsize = (3, 5, 10)
Tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(Tucker_orig);

ttf = NTFk.tlanalysis(T_orig, [csize...])

t = NTFk.analysis(T_orig, [csize], 1)


import PyCall
@PyCall.pyimport tensorly as tl
@PyCall.pyimport tensorly.decomposition as td

tf = PyCall.pyimport("tensorflow")
o = tf[:random_normal]((3,3,3))
o = tf[:random_normal]((10,20))[:numpy]()
tl.set_backend("tensorflow");
t = td.non_negative_tucker(o, rank=(2,2,2), n_iter_max=2)

Random.seed!(1)
csize = (2, 2, 2)
tsize = (3, 3, 3)
Tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(Tucker_orig);

@info("numpy")
tl.set_backend("numpy")
@time o = tl.backend[:tensor](T_orig)
@time t = td.non_negative_tucker(o, rank=csize, n_iter_max=2)

@info("mxnet")
tl.set_backend("mxnet")
@time o = tl.backend[:tensor](T_orig)
@time t = td.non_negative_tucker(o, rank=csize, n_iter_max=2)

@info("pytorch")
tl.set_backend("pytorch")
@time o = tl.backend[:tensor](T_orig)
@time t = td.non_negative_tucker(o, rank=csize, n_iter_max=2)

@info("tensorflow")
tl.set_backend("tensorflow");
@time o = tl.backend[:tensor](T_orig)
@time t = td.non_negative_tucker(o, rank=csize, n_iter_max=2)

Random.seed!(1)
csize = (10, 10, 10)
tsize = (1000, 1000, 1000)
Tucker_orig = NTFk.rand_tucker(csize, tsize, factors_nonneg=true, core_nonneg=true)
T_orig = TensorDecompositions.compose(Tucker_orig);

@info("numpy")
tl.set_backend("numpy")
@time o = tl.backend[:tensor](T_orig);
@time td.non_negative_tucker(o, rank=csize, n_iter_max=2);

@info("mxnet")
tl.set_backend("mxnet")
@time o = tl.backend[:tensor](T_orig);
@time td.non_negative_tucker(o, rank=csize, n_iter_max=2);

@info("pytorch")
tl.set_backend("pytorch")
@time o = tl.backend[:tensor](T_orig);
@time td.non_negative_tucker(o, rank=csize, n_iter_max=2);

@info("tensorflow")
tl.set_backend("tensorflow");
@time o = tl.backend[:tensor](T_orig);
@time td.non_negative_tucker(o, rank=csize, n_iter_max=2);

@info("numpy")
tl.set_backend("numpy")
@time td.non_negative_tucker(tl.backend[:tensor](T_orig), rank=csize, n_iter_max=2);

@info("mxnet")
tl.set_backend("mxnet")
@time td.non_negative_tucker(tl.backend[:tensor](T_orig), rank=csize, n_iter_max=2);

@info("pytorch")
tl.set_backend("pytorch")
@time td.non_negative_tucker(tl.backend[:tensor](T_orig), rank=csize, n_iter_max=2);

@info("tensorflow")
tl.set_backend("tensorflow");
@time td.non_negative_tucker(tl.backend[:tensor](T_orig), rank=csize, n_iter_max=2);