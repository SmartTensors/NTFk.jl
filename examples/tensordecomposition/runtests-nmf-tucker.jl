import NTFk

Random.seed!(2015)
a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
Q = [1; 2; 3]
X = W * H
tsize = (20, 5, 3)
T_orig = Array{Float64}(undef, tsize)
T_orig[:,:,1] = X
T_orig[:,:,2] = X * 2
T_orig[:,:,3] = X * 3

# T = add_noise(T_orig, 0.6, true)
T = T_orig

NTFk.analysis(T, [tsize])
sizes = [(1,1,1), (2,2,2), (2,2,1), (3,2,1), (2,3,1), (3,3,1), (3,3,2), (3,3,3)]
tsf, csize, ibest = NTFk.analysis(T, sizes, 10; ini_decomp=:hosvd, progressbar=true, tol=1e-6, max_iter=10000)

if ibest != 3
	@warn("Something might be wrong but most probably is not a big deal; the best results should be #3")
	@info("Estimated core size = $csize; best result is #$ibest; the correct core size should be (2,2,1)")
end

NTFk.plotcmptensors(T_orig, tsf[ibest], 3; progressbar=false)
@show cor(W[:,1], tsf[ibest].factors[1][:,1])
@show cor(W[:,2], tsf[ibest].factors[1][:,2])
@show cor(H[1,:], tsf[ibest].factors[2][:,1])
@show cor(H[2,:], tsf[ibest].factors[2][:,2])
@show cor(Q, tsf[ibest].factors[3][:,1])
p = Gadfly.plot(
	Gadfly.Guide.title("Signal 1"),
	Gadfly.layer(x=1:length(W[:,1]), y=W[:,1], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
	Gadfly.layer(x=1:length(W[:,1]), y=tsf[ibest].factors[1][:,1], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
display(p); println()
p = Gadfly.plot(
	Gadfly.Guide.title("Signal 2"),
	Gadfly.layer(x=1:length(W[:,2]), y=W[:,2], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
	Gadfly.layer(x=1:length(W[:,2]), y=tsf[ibest].factors[1][:,2], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
display(p); println()
p = Gadfly.plot(
	Gadfly.Guide.title("Mixer 1"),
	Gadfly.layer(x=1:length(H[1,:]), y=H[1,:], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
	Gadfly.layer(x=1:length(H[1,:]), y=tsf[ibest].factors[2][:,1], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
display(p); println()
p = Gadfly.plot(
	Gadfly.Guide.title("Mixer 2"),
	Gadfly.layer(x=1:length(H[2,:]), y=H[2,:], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
	Gadfly.layer(x=1:length(H[2,:]), y=tsf[ibest].factors[2][:,2], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
display(p); println()
p = Gadfly.plot(
	Gadfly.Guide.title("Time change"),
	Gadfly.layer(x=1:length(Q), y=Q, Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
	Gadfly.layer(x=1:length(Q), y=tsf[ibest].factors[3][:,1], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
display(p); println()