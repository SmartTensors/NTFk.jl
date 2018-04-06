import NTFk
import Base.Test

@Base.Test.testset "NTFk plot" begin
	r = reshape(repeat(collect(1/100:1/100:1), inner=100), (100,100))
	NTFk.plotmatrix(r; colormap=NTFk.colormap_hsv2);
	display(d); println()
	NTFk.plotmatrix(r; colormap=NTFk.colormap_hsv);
	display(d); println()
	@Base.Test.test isapprox(1, 1, atol=1e-6)
end