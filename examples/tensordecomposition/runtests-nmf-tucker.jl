import dNTF

srand(2015)
a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
Q = [1; 2; 3]
X = W * H
tsize = (20, 5, 3)
T_orig = Array{Float64}(tsize)
T_orig[:,:,1] = X
T_orig[:,:,2] = X * 2
T_orig[:,:,3] = X * 3

# T = add_noise(T_orig, 0.6, true)
T = T_orig

dNTF.analysis(T, [tsize])
sizes = [(1,1,1), (2,2,2), (2,2,1), (2,1,1), (2,1,2), (1,2,2), (3,2,1), (2,3,1), (3,3,1)]
cpf, csize, ibest = dNTF.analysis(T, sizes, 10; ini_decomp=nothing, progressbar=true, tol=1e-8, max_iter=1000)

if ibest == 2 || ibest == 3 # these should be the best results; otherwise the comparison fails
	dNTF.plotcmptensor(T_orig, T_esta[ibest], 3; progressbar=false)
	@show cor(W[:,1], tucker_spnn[ibest].factors[1][:,1])
	@show cor(W[:,2], tucker_spnn[ibest].factors[1][:,2])
	@show cor(H[1,:], tucker_spnn[ibest].factors[2][:,1])
	@show cor(H[2,:], tucker_spnn[ibest].factors[2][:,2])
	@show cor(Q, tucker_spnn[ibest].factors[3][:,1])
else
	warn("something is not correct; the best results should be #3 (or #2)")
end

if ibest != 3 # theoretically this should be the best result!!!
	ibest = 3
	dNTF.plotcmptensor(T_orig, T_esta[ibest], 3; progressbar=false)
	@show cor(W[:,1], tucker_spnn[ibest].factors[1][:,1])
	@show cor(W[:,2], tucker_spnn[ibest].factors[1][:,2])
	@show cor(H[1,:], tucker_spnn[ibest].factors[2][:,1])
	@show cor(H[2,:], tucker_spnn[ibest].factors[2][:,2])
	@show cor(Q, tucker_spnn[ibest].factors[3][:,1])
	p = Gadfly.plot(
		Gadfly.Guide.title("Signal 1"),
		Gadfly.layer(x=1:length(W[:,1]), y=W[:,1], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
		Gadfly.layer(x=1:length(W[:,1]), y=tucker_spnn[ibest].factors[1][:,1], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
	display(p); println()
	p = Gadfly.plot(
		Gadfly.Guide.title("Signal 2"),
		Gadfly.layer(x=1:length(W[:,2]), y=W[:,2], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
		Gadfly.layer(x=1:length(W[:,2]), y=tucker_spnn[ibest].factors[1][:,2], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
	display(p); println()
	p = Gadfly.plot(
		Gadfly.Guide.title("Mixer 1"),
		Gadfly.layer(x=1:length(H[1,:]), y=H[1,:], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
		Gadfly.layer(x=1:length(H[1,:]), y=tucker_spnn[ibest].factors[2][:,1], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
	display(p); println()
	p = Gadfly.plot(
		Gadfly.Guide.title("Mixer 2"),
		Gadfly.layer(x=1:length(H[2,:]), y=H[2,:], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
		Gadfly.layer(x=1:length(H[2,:]), y=tucker_spnn[ibest].factors[2][:,2], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
	display(p); println()
	p = Gadfly.plot(
		Gadfly.Guide.title("Time change"),
		Gadfly.layer(x=1:length(Q), y=Q, Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "blue"))),
		Gadfly.layer(x=1:length(Q), y=tucker_spnn[ibest].factors[3][:,1], Gadfly.Geom.line, Gadfly.Theme(default_color=parse(Colors.Colorant, "red"))))
	display(p); println()
end