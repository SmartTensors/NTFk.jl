import Gadfly
import Measures
import Colors
import Compose
import TensorToolbox
import TensorDecompositions
import Distributions

colors = ["red", "blue", "green", "orange", "magenta", "cyan", "brown", "pink", "lime", "navy", "maroon", "yellow", "olive", "springgreen", "teal", "coral", "lavender", "beige"]
ncolors = length(colors)

# r = reshape(repeat(collect(1/100:1/100:1), inner=100), (100,100))
# NTFk.plotmatrix(r; colormap=NTFk.colormap_hsv2);

rbwlong_ncar = [
0.250980 0.000000 0.276078;
0.250980 0.000000 0.326275;
0.250980 0.000000 0.376471;
0.250980 0.000000 0.426667;
0.250980 0.000000 0.476863;
0.250980 0.000000 0.527059;
0.250980 0.000000 0.577255;
0.250980 0.000000 0.627451;
0.250980 0.000000 0.677647;
0.250980 0.000000 0.727843;
0.238431 0.012549 0.765294;
0.213333 0.037647 0.790000;
0.188235 0.062745 0.814706;
0.163137 0.087843 0.839412;
0.138039 0.112941 0.864118;
0.112941 0.138039 0.888824;
0.087843 0.163137 0.913529;
0.062745 0.188235 0.938235;
0.037647 0.213333 0.962941;
0.012549 0.238431 0.987647;
0.000000 0.263529 1.000000;
0.000000 0.288627 1.000000;
0.000000 0.313725 1.000000;
0.000000 0.338824 1.000000;
0.000000 0.363922 1.000000;
0.000000 0.389020 1.000000;
0.000000 0.414118 1.000000;
0.000000 0.439216 1.000000;
0.000000 0.464314 1.000000;
0.000000 0.489412 1.000000;
0.000000 0.508235 1.000000;
0.000000 0.520784 1.000000;
0.000000 0.533333 1.000000;
0.000000 0.545882 1.000000;
0.000000 0.558431 1.000000;
0.000000 0.570980 1.000000;
0.000000 0.583529 1.000000;
0.000000 0.596078 1.000000;
0.000000 0.608627 1.000000;
0.000000 0.621176 1.000000;
0.012549 0.633725 1.000000;
0.037647 0.646275 1.000000;
0.062745 0.658824 1.000000;
0.087843 0.671373 1.000000;
0.112941 0.683922 1.000000;
0.138039 0.696471 1.000000;
0.163137 0.709020 1.000000;
0.188235 0.721569 1.000000;
0.213333 0.734118 1.000000;
0.238431 0.746667 1.000000;
0.250980 0.759216 1.000000;
0.250980 0.771765 1.000000;
0.250980 0.784314 1.000000;
0.250980 0.796863 1.000000;
0.250980 0.809412 1.000000;
0.250980 0.821961 1.000000;
0.250980 0.834510 1.000000;
0.250980 0.847059 1.000000;
0.250980 0.859608 1.000000;
0.250980 0.872157 1.000000;
0.250980 0.884510 1.000000;
0.250980 0.896667 1.000000;
0.250980 0.908824 1.000000;
0.250980 0.920980 1.000000;
0.250980 0.933137 1.000000;
0.250980 0.945294 1.000000;
0.250980 0.957451 1.000000;
0.250980 0.969608 1.000000;
0.250980 0.981765 1.000000;
0.250980 0.993922 1.000000;
0.250980 1.000000 0.987647;
0.250980 1.000000 0.962941;
0.250980 1.000000 0.938235;
0.250980 1.000000 0.913529;
0.250980 1.000000 0.888824;
0.250980 1.000000 0.864118;
0.250980 1.000000 0.839412;
0.250980 1.000000 0.814706;
0.250980 1.000000 0.790000;
0.250980 1.000000 0.765294;
0.250980 1.000000 0.727843;
0.250980 1.000000 0.677647;
0.250980 1.000000 0.627451;
0.250980 1.000000 0.577255;
0.250980 1.000000 0.527059;
0.250980 1.000000 0.476863;
0.250980 1.000000 0.426667;
0.250980 1.000000 0.376471;
0.250980 1.000000 0.326275;
0.250980 1.000000 0.276078;
0.263529 1.000000 0.250980;
0.288627 1.000000 0.250980;
0.313725 1.000000 0.250980;
0.338824 1.000000 0.250980;
0.363922 1.000000 0.250980;
0.389020 1.000000 0.250980;
0.414118 1.000000 0.250980;
0.439216 1.000000 0.250980;
0.464314 1.000000 0.250980;
0.489412 1.000000 0.250980;
0.514510 1.000000 0.250980;
0.539608 1.000000 0.250980;
0.564706 1.000000 0.250980;
0.589804 1.000000 0.250980;
0.614902 1.000000 0.250980;
0.640000 1.000000 0.250980;
0.665098 1.000000 0.250980;
0.690196 1.000000 0.250980;
0.715294 1.000000 0.250980;
0.740392 1.000000 0.250980;
0.765294 1.000000 0.250980;
0.790000 1.000000 0.250980;
0.814706 1.000000 0.250980;
0.839412 1.000000 0.250980;
0.864118 1.000000 0.250980;
0.888824 1.000000 0.250980;
0.913529 1.000000 0.250980;
0.938235 1.000000 0.250980;
0.962941 1.000000 0.250980;
0.987647 1.000000 0.250980;
1.000000 0.993922 0.250980;
1.000000 0.981765 0.250980;
1.000000 0.969608 0.250980;
1.000000 0.957451 0.250980;
1.000000 0.945294 0.250980;
1.000000 0.933137 0.250980;
1.000000 0.920980 0.250980;
1.000000 0.908824 0.250980;
1.000000 0.896667 0.250980;
1.000000 0.884510 0.250980;
1.000000 0.865882 0.250980;
1.000000 0.840784 0.250980;
1.000000 0.815686 0.250980;
1.000000 0.790588 0.250980;
1.000000 0.765490 0.250980;
1.000000 0.740392 0.250980;
1.000000 0.715294 0.250980;
1.000000 0.690196 0.250980;
1.000000 0.665098 0.250980;
1.000000 0.640000 0.250980;
1.000000 0.614902 0.250980;
1.000000 0.589804 0.250980;
1.000000 0.564706 0.250980;
1.000000 0.539608 0.250980;
1.000000 0.514510 0.250980;
1.000000 0.489412 0.250980;
1.000000 0.464314 0.250980;
1.000000 0.439216 0.250980;
1.000000 0.414118 0.250980;
1.000000 0.389020 0.250980;
1.000000 0.363922 0.250980;
1.000000 0.338824 0.250980;
1.000000 0.313725 0.250980;
1.000000 0.288627 0.250980;
1.000000 0.263529 0.250980;
1.000000 0.238431 0.250980;
1.000000 0.213333 0.250980;
1.000000 0.188235 0.250980;
1.000000 0.163137 0.250980;
1.000000 0.138039 0.250980;
1.000000 0.138039 0.276078;
1.000000 0.163137 0.326275;
1.000000 0.188235 0.376471;
1.000000 0.213333 0.426667;
1.000000 0.238431 0.476863;
1.000000 0.263529 0.527059;
1.000000 0.288627 0.577255;
1.000000 0.313725 0.627451;
1.000000 0.338824 0.677647;
1.000000 0.363922 0.727843;
1.000000 0.389020 0.765294;
1.000000 0.414118 0.790000;
1.000000 0.439216 0.814706;
1.000000 0.464314 0.839412;
1.000000 0.489412 0.864118;
1.000000 0.514510 0.888824;
1.000000 0.539608 0.913529;
1.000000 0.564706 0.938235;
1.000000 0.589804 0.962941;
1.000000 0.614902 0.987647;
1.000000 0.640000 0.994118;
1.000000 0.665098 0.982353;
1.000000 0.690196 0.970588;
1.000000 0.715294 0.958824;
1.000000 0.740392 0.947059;
1.000000 0.765490 0.935294;
1.000000 0.790588 0.923529;
1.000000 0.815686 0.911765;
1.000000 0.840784 0.900000;
1.000000 0.865882 0.888235;
1.000000 0.883343 1.000000;
1.000000 0.895623 1.000000;
1.000000 0.907903 1.000000;
1.000000 0.920182 1.000000;
1.000000 0.932462 1.000000;
1.000000 0.944742 1.000000;
1.000000 0.957021 1.000000;
1.000000 0.969301 1.000000;
1.000000 0.981581 1.000000;
1.000000 0.993860 1.000000;
];


rgb_ncar = [
	 # 4   0   3;
	 # 9   0   7;
	# 13   0  10;
	18   0  14;
	22   0  19;
	27   0  23;
	31   0  28;
	36   0  32;
	40   0  38;
	45   0  43;
	50   0  48;
	58   0  59;
	64   0  68;
	68   0  72;
	69   0  77;
	72   0  81;
	74   0  86;
	77   0  91;
	79   0  95;
	80   0 100;
	82   0 104;
	83   0 109;
	84   0 118;
	86   0 122;
	88   0 132;
	86   0 136;
	87   0 141;
	87   0 145;
	87   0 150;
	85   0 154;
	84   0 159;
	84   0 163;
	84   0 168;
	79   0 177;
	78   0 182;
	77   0 186;
	76   0 191;
	70   0 200;
	68   0 204;
	66   0 209;
	60   0 214;
	58   0 218;
	55   0 223;
	46   0 232;
	43   0 236;
	40   0 241;
	36   0 245;
	33   0 250;
	25   0 255;
	16   0 255;
	12   0 255;
	 4   0 255;
	 0   0 255;
	 0   4 255;
	 0  16 255;
	 0  21 255;
	 0  25 255;
	 0  29 255;
	 0  38 255;
	 0  42 255;
	 0  46 255;
	 0  51 255;
	 0  63 255;
	 0  67 255;
	 0  72 255;
	 0  84 255;
	 0  89 255;
	 0  93 255;
	 0  97 255;
	 0 106 255;
	 0 110 255;
	 0 114 255;
	 0 119 255;
	 0 127 255;
	 0 135 255;
	 0 140 255;
	 0 152 255;
	 0 157 255;
	 0 161 255;
	 0 165 255;
	 0 174 255;
	 0 178 255;
	 0 182 255;
	 0 187 255;
	 0 195 255;
	 0 199 255;
	 0 216 255;
	 0 220 255;
	 0 225 255;
	 0 229 255;
	 0 233 255;
	 0 242 255;
	 0 246 255;
	 0 250 255;
	 0 255 255;
	 0 255 246;
	 0 255 242;
	 0 255 238;
	 0 255 225;
	 0 255 216;
	 0 255 212;
	 0 255 203;
	 0 255 199;
	 0 255 195;
	 0 255 191;
	 0 255 187;
	 0 255 178;
	 0 255 174;
	 0 255 170;
	 0 255 157;
	 0 255 152;
	 0 255 144;
	 0 255 135;
	 0 255 131;
	 0 255 127;
	 0 255 123;
	 0 255 114;
	 0 255 110;
	 0 255 106;
	 0 255 102;
	 0 255  89;
	 0 255  84;
	 0 255  80;
	 0 255  76;
	 0 255  63;
	 0 255  59;
	 0 255  55;
	 0 255  46;
	 0 255  42;
	 0 255  38;
	 0 255  25;
	 0 255  21;
	 0 255  16;
	 0 255  12;
	 0 255   8;
	 0 255   0;
	 8 255   0;
	12 255   0;
	21 255   0;
	25 255   0;
	29 255   0;
	42 255   0;
	46 255   0;
	51 255   0;
	55 255   0;
	63 255   0;
	67 255   0;
	72 255   0;
	76 255   0;
	89 255   0;
	93 255   0;
	97 255   0;
 110 255   0;
 114 255   0;
 119 255   0;
 123 255   0;
 131 255   0;
 135 255   0;
 140 255   0;
 144 255   0;
 153 255   0;
 161 255   0;
 165 255   0;
 178 255   0;
 182 255   0;
 187 255   0;
 191 255   0;
 199 255   0;
 203 255   0;
 208 255   0;
 212 255   0;
 221 255   0;
 225 255   0;
 242 255   0;
 246 255   0;
 250 255   0;
 255 255   0;
 255 250   0;
 255 242   0;
 255 238   0;
 255 233   0;
 255 229   0;
 255 221   0;
 255 216   0;
 255 212   0;
 255 199   0;
 255 191   0;
 255 187   0;
 255 178   0;
 255 174   0;
 255 170   0;
 255 165   0;
 255 161   0;
 255 153   0;
 255 148   0;
 255 144   0;
 255 131   0;
 255 127   0;
 255 119   0;
 255 110   0;
 255 106   0;
 255 102   0;
 255  97   0;
 255  89   0;
 255  85   0;
 255  80   0;
 255  76   0;
 255  63   0;
 255  59   0;
 255  55   0;
 255  51   0;
 255  38   0;
 255  34   0;
 255  29   0;
 255  21   0;
 255  17   0;
 255  12   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0;
 255   0   0];

gist_ncar = [0  0 128;
	 0   9 115;
	 0  19 103;
	 0  28  91;
	 0  38  79;
	 0  47  66;
	 0  57  54;
	 0  66  42;
	 0  76  30;
	 0  85  18;
	 0  95   5;
	 0  94  12;
	 0  85  37;
	 0  75  61;
	 0  66  85;
	 0  56 110;
	 0  47 134;
	 0  37 158;
	 0  28 183;
	 0  18 207;
	 0   9 231;
	 0   0 255;
	 0  19 255;
	 0  37 255;
	 0  55 255;
	 0  73 255;
	 0  92 255;
	 0 110 255;
	 0 128 255;
	 0 146 255;
	 0 165 255;
	 0 183 255;
	 0 194 255;
	 0 200 255;
	 0 206 255;
	 0 212 255;
	 0 218 255;
	 0 225 255;
	 0 231 255;
	 0 237 255;
	 0 243 255;
	 0 249 255;
	 0 254 253;
	 0 254 244;
	 0 253 234;
	 0 253 225;
	 0 253 215;
	 0 252 205;
	 0 252 196;
	 0 251 186;
	 0 251 176;
	 0 250 167;
	 0 250 157;
	 0 250 144;
	 0 250 130;
	 0 251 115;
	 0 251 100;
	 0 252  85;
	 0 252  71;
	 0 253  56;
	 0 253  41;
	 0 254  27;
	 0 254  12;
	 1 254   0;
	11 249   0;
	21 244   0;
	30 239   0;
	40 235   0;
	50 230   0;
	59 225   0;
	69 220   0;
	79 216   0;
	89 211   0;
	98 206   0;
 103 208   0;
 106 213   0;
 108 217   0;
 110 222   0;
 113 227   0;
 115 232   0;
 117 236   0;
 120 241   0;
 122 246   0;
 125 251   0;
 128 255   1;
 134 255   7;
 140 255  13;
 146 255  19;
 153 255  24;
 159 255  30;
 165 255  36;
 171 255  42;
 177 255  48;
 184 255  54;
 190 255  60;
 196 255  57;
 202 255  51;
 208 255  45;
 214 255  39;
 220 255  33;
 226 255  28;
 232 255  22;
 238 255  16;
 244 255  10;
 250 255   4;
 255 253   0;
 255 250   0;
 255 246   0;
 255 242   0;
 255 238   0;
 255 234   0;
 255 231   0;
 255 227   0;
 255 223   0;
 255 219   0;
 255 215   0;
 255 212   1;
 255 209   2;
 255 207   3;
 255 204   5;
 255 201   6;
 255 198   8;
 255 195   9;
 255 192  11;
 255 189  12;
 255 186  14;
 255 181  14;
 255 170  13;
 255 159  11;
 255 148  10;
 255 137   8;
 255 126   7;
 255 115   5;
 255 103   4;
 255  92   3;
 255  81   1;
 255  70   0;
 255  63   0;
 255  56   0;
 255  50   0;
 255  43   0;
 255  37   0;
 255  30   0;
 255  23   0;
 255  17   0;
 255  10   0;
 255   4   0;
 255   0   8;
 255   0  33;
 255   0  57;
 255   0  82;
 255   0 106;
 255   0 130;
 255   0 155;
 255   0 179;
 255   0 203;
 255   0 228;
 255   0 252;
 246   4 255;
 236   8 255;
 227  13 255;
 217  17 255;
 208  22 255;
 198  27 255;
 189  31 255;
 179  36 255;
 170  40 255;
 160  45 255;
 158  51 254;
 166  59 252;
 174  66 251;
 182  74 249;
 190  82 247;
 197  90 246;
 205  98 244;
 213 106 242;
 221 113 241;
 229 121 239;
 237 129 238;
 238 135 238;
 239 141 239;
 239 146 239;
 240 152 240;
 241 158 241;
 241 164 241;
 242 169 242;
 243 175 243;
 243 181 243;
 244 186 244;
 245 192 245;
 246 199 246;
 247 205 247;
 248 211 248;
 249 217 249;
 250 223 250;
 251 230 251;
 252 236 252;
 253 242 253;
 254 248 254;
 255 255 255];

colormap_rbwlong = [Gadfly.Scale.lab_gradient([Colors.RGB{Colors.N0f8}(rbwlong_ncar[i, :]...) for i=1:size(rbwlong_ncar, 1)]...)]
colormap_ncar = [Gadfly.Scale.lab_gradient([Colors.RGB{Colors.N0f8}(rgb_ncar[i, :]./255...) for i=1:size(rgb_ncar, 1)]...)]
colormap_gist = [Gadfly.Scale.lab_gradient([Colors.RGB{Colors.N0f8}(gist_ncar[i, :]./255...) for i=1:size(gist_ncar, 1)]...)]
colormap_hsv2 = [Gadfly.Scale.lab_gradient(Colors.RGB{Colors.N0f8}(42/255, 28/255, 14/255), parse(Colors.Colorant, "coral"), parse(Colors.Colorant, "darkmagenta"), parse(Colors.Colorant, "peachpuff"), parse(Colors.Colorant, "darkblue"), parse(Colors.Colorant, "cyan"), parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"))]
colormap_hsv = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "magenta"), parse(Colors.Colorant, "peachpuff"), parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "cyan"), parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"))]
colormap_rbw2 = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "cyan"), parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"), parse(Colors.Colorant, "darkmagenta"))]
colormap_rbw = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "cyan"), parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"))]
colormap_gyr = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red"))]
colormap_gy = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"))]
colormap_wb = [Gadfly.Scale.lab_gradient(parse(Colors.Colorant, "white"), parse(Colors.Colorant, "black"))]
# colormap_g_ = [Gadfly.Scale.lab_gradient(Colors.RGBA{Colors.N0f8}(0,1,0,0), Colors.RGBA{Colors.N0f8}(0,1,0,1))]

function plot2dtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true,  datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude), filter=vec(1:length(order)), xmin=datestart, xmax=dateend, transform=nothing, linewidth=2Gadfly.pt, separate::Bool=false)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	if datestart != nothing
		if dateend == nothing
			xvalues = datestart .+ vec(collect(eval(parse(dateincrement))(0):eval(parse(dateincrement))(1):eval(parse(dateincrement))(nx-1)))
		else
			xvalues = datestart .+ (vec(collect(1:nx)) ./ nx .* (dateend .- datestart))
		end
	end
	ncomponents = length(filter)
	loopcolors = ncomponents > ncolors ? true : false
	# if loopcolors
	# 	colorloops = convert(Int64, floor(ncomponents / ncolors))
	# end
	componentnames = map(i->"T$i", filter)
	p = t.factors[dim]
	if transform != nothing
		p = transform.(p)
	end
	pl = Vector{Any}(ncomponents)
	for i = 1:ncomponents
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=p[:, order[filter[i]]], Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=cc))
	end
	tx = timescale ? [] : [Gadfly.Coord.Cartesian(xmin=minimum(xvalues), xmax=maximum(xvalues))]
	# @show [repeat(colors, colorloops); colors[1:(ncomponents - colorloops * ncolors)]]
	# tc = loopcolors ? [Gadfly.Guide.manual_color_key("", componentnames, [repeat(colors, colorloops); colors[1:(ncomponents - colorloops * ncolors)]])] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:ncomponents])] # this does not work
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:ncomponents])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	xm = (xmin == nothing && xmax == nothing) ? [] : [Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., tm..., xm...]
	end
	if separate
		for i = 1:ncomponents
			tt = title == "" ? title : title * " Signal $(order[filter[i]])"
			ff = Gadfly.plot(Gadfly.layer(x=xvalues, y=p[:, order[filter[i]]], Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=parse(Colors.Colorant, "red"))), Gadfly.Guide.title(tt), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., tx..., xm...)
			!quiet && (display(ff); println())
			fs = split(filename, ".")
			fn = fs[1] * "-$(lpad(order[filter[i]],4,0))." * fs[2]
			Gadfly.draw(Gadfly.PNG(joinpath(figuredir, fn), hsize, vsize, dpi=150), ff)
		end
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc..., tm..., tx..., xm...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plot2dmodtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, functionname::String="mean"; quiet::Bool=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude), xmin=datestart, xmax=dateend)
	recursivemkdir(figuredir; filename=false, transform=nothing)
	recursivemkdir(filename)
	csize = TensorToolbox.mrank(t.core)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	if datestart != nothing
		if dateend == nothing
			xvalues = datestart .+ vec(collect(eval(parse(dateincrement))(0):eval(parse(dateincrement))(1):eval(parse(dateincrement))(nx-1)))
		else
			xvalues = datestart .+ (vec(collect(1:nx)) ./nx .* (dateend .- datestart))
		end
	end
	componentnames = map(i->"T$i", 1:crank)
	dp = Vector{Int64}(0)
	for i = 1:ndimensons
		if i != dim
			push!(dp, i)
		end
	end
	pl = Vector{Any}(crank)
	tt = deepcopy(t)
	for (i, o) = enumerate(order)
		for j = 1:crank
			if o !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		X2 = TensorDecompositions.compose(tt)
		tt.core .= t.core
		tm = eval(parse(functionname))(X2, dp)
		if transform != nothing
			tm = transform.(tm)
		end
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
	end
	tx = timescale ? [] : [Gadfly.Coord.Cartesian(xmin=minimum(xvalues), xmax=maximum(xvalues))]
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:crank])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	xm = (xmin == nothing && xmax == nothing) ? [] : [Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., xm..., tc...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., xm..., tc..., tx...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plot2dmodtensorcomponents(X::Array, t::TensorDecompositions.Tucker, dim::Integer=1, functionname1::String="mean", functionname2::String="mean"; quiet=false, hsize=8Compose.inch, vsize=4Compose.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", code::Bool=false, order=gettensorcomponentorder(t, dim; method=:factormagnitude), xmin=datestart, xmax=dateend, transform=nothing)
	csize = TensorToolbox.mrank(t.core)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	crank = csize[dim]
	loopcolors = crank > ncolors ? true : false
	nx, ny = size(t.factors[dim])
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	if datestart != nothing
		if dateend == nothing
			xvalues = datestart .+ vec(collect(eval(parse(dateincrement))(0):eval(parse(dateincrement))(1):eval(parse(dateincrement))(nx-1)))
		else
			xvalues = datestart .+ (vec(collect(1:nx)) ./nx .* (dateend .- datestart))
		end
	end
	componentnames = map(i->"T$i", 1:crank)
	dp = Vector{Int64}(0)
	for i = 1:ndimensons
		if i != dim
			push!(dp, i)
		end
	end
	pl = Vector{Any}(crank+2)
	tt = deepcopy(t)
	for (i, o) = enumerate(order)
		for j = 1:crank
			if o !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		X2 = TensorDecompositions.compose(tt)
		tt.core .= t.core
		tm = eval(parse(functionname1))(X2, dp)
		if transform != nothing
			tm = transform.(tm)
		end
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		pl[i] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
	end
	tm = map(j->eval(parse(functionname2))(vec(X[ntuple(k->(k == dim ? j : Colon()), ndimensons)...])), 1:nx)
	pl[crank+1] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=3Gadfly.pt, line_style=:dot, default_color=parse(Colors.Colorant, "gray")))
	Xe = TensorDecompositions.compose(t)
	tm = map(j->eval(parse(functionname2))(vec(Xe[ntuple(k->(k == dim ? j : Colon()), ndimensons)...])), 1:nx)
	pl[crank+2] = Gadfly.layer(x=xvalues, y=tm, Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=parse(Colors.Colorant, "gray85")))
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", [componentnames; "Est."; "True"], [colors[1:crank]; "gray85"; "gray"])]
	tm = (ymin == nothing && ymax == nothing) ? [] : [Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax)]
	xm = (xmin == nothing && xmax == nothing) ? [] : [Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax)]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., xm..., tc...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tm..., xm..., tc...)
	!quiet && (display(ff); println())
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=150), ff)
	end
	return ff
end

function plotmatrix(X::AbstractVector; kw...)
	plotmatrix(convert(Array{Float64,2}, permutedims(X)); kw...)
end

function plotmatrix(X::AbstractMatrix; minvalue=minimumnan(X), maxvalue=maximumnan(X), label="", title="", xlabel="", ylabel="", xticks=nothing, yticks=nothing, gm=[Gadfly.Guide.xticks(label=false, ticks=nothing), Gadfly.Guide.yticks(label=false, ticks=nothing)], masize::Int64=0, colormap=colormap_gyr, filename::String="", hsize=6Compose.inch, vsize=6Compose.inch, figuredir::String=".", colorkey::Bool=true, mask=nothing, polygon=nothing, contour=nothing, linewidth::Measures.Length{:mm,Float64}=1Gadfly.pt, linecolor="gray", transform=nothing)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	Xp = deepcopy(min.(max.(movingaverage(X, masize), minvalue), maxvalue))
	if transform != nothing
		Xp = transform.(Xp)
	end
	nanmask!(Xp, mask)
	if xticks != nothing
		gm = [gm..., Gadfly.Scale.x_discrete(labels=i->xticks[i]), Gadfly.Guide.xticks(label=true)]
	end
	if yticks != nothing
		gm = [gm..., Gadfly.Scale.y_discrete(labels=i->yticks[i]), Gadfly.Guide.yticks(label=true)]
	end
	cs = colorkey ? [] : [Gadfly.Theme(key_position = :none)]
	ds = min.(size(Xp)) == 1 ? [Gadfly.Scale.x_discrete, Gadfly.Scale.y_discrete] : []
	if polygon == nothing && contour == nothing
		p = Gadfly.spy(Xp, Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Guide.ColorKey(title=label), ds..., Gadfly.Scale.ContinuousColorScale(colormap..., minvalue=minvalue, maxvalue=maxvalue), Gadfly.Theme(major_label_font_size=24Gadfly.pt, key_label_font_size=12Gadfly.pt, bar_spacing=0Gadfly.mm), cs..., gm...)
	else
		is, js, values = Gadfly._findnz(x->!isnan(x), X)
		n, m = size(X)
		if polygon != nothing
			p = Gadfly.plot(Gadfly.layer(x=polygon[1], y=polygon[2], Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth, default_color=linecolor)), Gadfly.layer(x=js, y=is, color=values, Gadfly.Geom.rectbin()), Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Guide.ColorKey(title=label), ds..., Gadfly.Scale.ContinuousColorScale(colormap..., minvalue=minvalue, maxvalue=maxvalue), Gadfly.Theme(major_label_font_size=24Gadfly.pt, key_label_font_size=12Gadfly.pt, bar_spacing=0Gadfly.mm), cs..., gm..., Gadfly.Scale.x_continuous, Gadfly.Scale.y_continuous, Gadfly.Coord.cartesian(yflip=true, fixed=true, xmin=0.5, xmax=m+.5, ymin=0.5, ymax=n+.5))
		else
			p = Gadfly.plot(Gadfly.layer(z=permutedims(contour .* (maxvalue - minvalue) .+ minvalue), x=collect(1:size(contour, 2)), y=collect(1:size(contour, 1)), Gadfly.Geom.contour(levels=[minvalue]), Gadfly.Theme(line_width=linewidth, default_color=linecolor)), Gadfly.layer(x=js, y=is, color=values, Gadfly.Geom.rectbin()), Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Guide.ColorKey(title=label), ds..., Gadfly.Scale.ContinuousColorScale(colormap..., minvalue=minvalue, maxvalue=maxvalue), Gadfly.Theme(major_label_font_size=24Gadfly.pt, key_label_font_size=12Gadfly.pt, bar_spacing=0Gadfly.mm), cs..., gm..., Gadfly.Scale.x_continuous, Gadfly.Scale.y_continuous, Gadfly.Coord.cartesian(yflip=true, fixed=true, xmin=0.5, xmax=m+.5, ymin=0.5, ymax=n+.5))
		end
	end
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=300), p)
	end
	return p
end

function plotfactor(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1, cutoff::Number=0; kw...)
	plotmatrix(getfactor(t, dim, cutoff); kw...)
end

function plotfactors(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, cutoff::Number=0; prefix="", kw...)
	recursivemkdir(prefix)
	for i = 1:length(t.factors)
		display(plotfactor(t, i, cutoff; filename="$(prefix)_factor$(i).png", kw...))
		println()
	end
end

function plotcore(t::TensorDecompositions.Tucker, dim::Integer=1, cutoff::Number=0; kw...)
	plottensor(t.core, dim; progressbar=nothing, cutoff=true, cutvalue=cutoff, kw...)
end

function plottensor(t::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; mask=nothing, transform=nothing, kw...)
	X = TensorDecompositions.compose(t)
	if transform != nothing
		X = transform.(X)
	end
	if typeof(mask) <: Number
		nanmask!(X, mask)
	else
		nanmask!(X, mask, dim)
	end
	plottensor(X, dim; kw...)
end

function plottensor(X::AbstractArray{T,N}, dim::Integer=1; mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=minimumnan(X), maxvalue=maximumnan(X), prefix::String="", keyword="frame", movie::Bool=false, title="", hsize=6Compose.inch, vsize=6Compose.inch, moviedir::String=".", quiet::Bool=false, cleanup::Bool=true, sizes=size(X), timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateend=(datestart != nothing) ? datestart + eval(parse(dateincrement))(sizes[dim]) : nothing, dateincrement::String="Dates.Day", progressbar=progressbar_regular, colormap=colormap_gyr, cutoff::Bool=false, cutvalue::Number=0, vspeed=1.0, kw...) where {T,N}
	if !checkdimension(dim, N)
		return
	end
	recursivemkdir(moviedir; filename=false)
	recursivemkdir(prefix)
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		M = X[nt...]
		if cutoff && maximumnan(M) .<= cutvalue
			continue
		end
		g = plotmatrix(M; minvalue=minvalue, maxvalue=maxvalue, title=title, colormap=colormap, kw...)
		if progressbar != nothing
			f = progressbar(i, timescale, timestep, datestart, dateend, dateincrement)
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		!quiet && ((sizes[dim] > 1) && println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(g, f)); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Compose.vstack(g, f))
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-$(keyword)%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -filter:v "setpts=$vspeed*PTS" -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		if moviedir == "."
			moviedir, prefix = splitdir(prefix)
			if moviedir == ""
				moviedir = "."
			end
		end
		cleanup && run(`find $moviedir -name $prefix-$(keyword)"*".png -delete`)
	end
end

function zerotensorcomponents!(t::TensorDecompositions.Tucker, dim::Int)
	ndimensons = length(size(t.core))
	nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
	t.core[nt...] .= 0
end

function zerotensorcomponents!(t::TensorDecompositions.CANDECOMP, dim::Int)
	ndimensons = length(size(t.core))
	nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
	t.lambdas[nt...] .= 0
end

function namedimension(ndimensons::Int; char="C", names=("T", "X", "Y"))
	if ndimensons <= 3
		dimname = names
	else
		dimname = ntuple(i->"$char$i", ndimensons)
	end
	return dimname
end

function plottensorcomponents(X1::Array, t2::TensorDecompositions.CANDECOMP; prefix::String="", filter=(), kw...)
	recursivemkdir(prefix)
	ndimensons = length(size(X1))
	crank = length(t2.lambdas)
	tt = deepcopy(t2)
	for i = 1:crank
		info("Making component $i movie ...")
		tt.lambdas[1:end .!== i] = 0
		if length(filter) == 0
			X2 = TensorDecompositions.compose(tt)
		else
			X2 = TensorDecompositions.compose(tt)[filter...]
		end
		tt.lambdas .= t2.lambdas
		plot2tensors(X1, X2; progressbar=nothing, prefix=prefix * string(i), kw...)
	end
end

function plottensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t2.core), prefix::String="", filter=(), kw...)
	recursivemkdir(prefix)
	ndimensons = length(size(X1))
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	pt = getptdimensions(pdim, ndimensons, transpose)
	tt = deepcopy(t2)
	for i = 1:crank
		info("Making component $(dimname[dim])-$i movie ...")
		for j = 1:crank
			if i !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		if length(filter) == 0
			X2 = TensorDecompositions.compose(tt)
		else
			X2 = TensorDecompositions.compose(tt)[filter...]
		end
		tt.core .= t2.core
		title = pdim > 1 ? "$(dimname[dim])-$i" : ""
		plot2tensors(permutedims(X1, pt), permutedims(X2, pt), 1; progressbar=nothing, title=title, prefix=prefix * string(i), kw...)
	end
end

function plot2tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), mask=nothing, transform=nothing, prefix::String="", filter=(), order=gettensorcomponentorder(t, dim; method=:factormagnitude), kw...)
	recursivemkdir(prefix)
	ndimensons = length(csize)
	@assert dim >= 1 && dim <= ndimensons
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = getptdimensions(pdim, ndimensons, transpose)
	tt = deepcopy(t)
	X = Vector{Any}(crank)
	for i = 1:crank
		info("Making component $(dimname[dim])-$i movie ...")
		for j = 1:crank
			if i !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		if length(filter) == 0
			X[i] = TensorDecompositions.compose(tt)
		else
			X[i] = TensorDecompositions.compose(tt)[filter...]
		end
		nanmask!(X[i], mask)
		if transform != nothing
			X[i] = transform.(X[i])
		end
		tt.core .= t.core
	end
	plot2tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt), 1; prefix=prefix, kw...)
end

function plot2tensorcomponents(X1::Array, t2::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t2.core), mask=nothing, transform=nothing, prefix::String="", filter=(), order=gettensorcomponentorder(t, dim; method=:factormagnitude), kw...)
	recursivemkdir(prefix)
	ndimensons = length(size(X1))
	@assert dim >= 1 && dim <= ndimensons
	@assert ndimensons == length(csize)
	dimname = namedimension(ndimensons)
	crank = csize[dim]
	@assert crank > 1
	pt = getptdimensions(pdim, ndimensons, transpose)
	tt = deepcopy(t2)
	X2 = Vector{Any}(crank)
	for i = 1:crank
		info("Making component $(dimname[dim])-$i movie ...")
		for j = 1:crank
			if i !== j
				nt = ntuple(k->(k == dim ? j : Colon()), ndimensons)
				tt.core[nt...] .= 0
			end
		end
		if length(filter) == 0
			X2[i] = TensorDecompositions.compose(tt)
		else
			X2[i] = TensorDecompositions.compose(tt)[filter...]
		end
		if transform != nothing
			X2[i] = transform.(X2[i])
		end
		nanmask!(X2[i], mask)
		tt.core .= t2.core
	end
	plot3tensors(permutedims(X1, pt), permutedims(X2[order[1]], pt), permutedims(X2[order[2]], pt), 1; prefix=prefix, kw...)
end

function plottensorandcomponents(X::Array, t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; csize::Tuple=TensorToolbox.mrank(t.core), sizes=size(X), xtitle="Time", ytitle="Magnitude", timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateend=(datestart != nothing) ? datestart + eval(parse(dateincrement))(sizes[dim]) : nothing, dateincrement::String="Dates.Day", sscleanup::Bool=true, movie::Bool=false, moviedir=".", prefix::String="", keyword="frame", title="", quiet::Bool=false, filter=(), minvalue=minimumnan(X), maxvalue=maximumnan(X), hsize=12Compose.inch, vsize=12Compose.inch, colormap=colormap_gyr, functionname="mean", vspeed=1.0, transform=nothing, transform2d=nothing, mask=nothing, kw...)
	ndimensons = length(sizes)
	if !checkdimension(dim, ndimensons) || !checkdimension(pdim, ndimensons)
		return
	end
	recursivemkdir(moviedir; filename=false)
	recursivemkdir(prefix)
	dimname = namedimension(ndimensons; char="D", names=("Row", "Column", "Layer"))
	s2 = plot2dmodtensorcomponents(X, t, dim, functionname; xtitle=xtitle, ytitle=ytitle, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, timescale=timescale, quiet=true, code=true, transform=transform2d)
	progressbar_2d = make_progressbar_2d(s2)
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : Colon()), ndimensons)
		p1 = plotmatrix(X[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=title, colormap=colormap, transform=transform, mask=mask)
		p2 = progressbar_2d(i, timescale, timestep, datestart, dateend, dateincrement)
		!quiet && (sizes[dim] > 1) && (println(framename); Gadfly.draw(Gadfly.PNG(hsize, vsize, dpi=150), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 2/3), Gadfly.render(p1)), Compose.compose(Compose.context(0, 0, 1, 1/3), Gadfly.render(p2)))); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 2/3), Gadfly.render(p1)), Compose.compose(Compose.context(0, 0, 1, 1/3), Gadfly.render(p2))))
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-$(keyword)%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -filter:v "setpts=$vspeed*PTS" -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		if moviedir == "."
			moviedir, prefix = splitdir(prefix)
			if moviedir == ""
				moviedir = "."
			end
		end
		cleanup && run(`find $moviedir -name $prefix-$(keyword)"*".png -delete`)
	end
end

function plot3tensorsandcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; xtitle="Time", ytitle="Magnitude", timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", functionname="mean", order=gettensorcomponentorder(t, dim; method=:factormagnitude), filter=vec(1:length(order)), xmin=datestart, xmax=dateend, ymin=nothing, ymax=nothing, transform2d=nothing, kw...)
	ndimensons = length(t.factors)
	if !checkdimension(dim, ndimensons) || !checkdimension(pdim, ndimensons)
		return
	end
	s2 = plot2dtensorcomponents(t, dim; xtitle=xtitle, ytitle=ytitle, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=true, code=true, order=order, filter=filter, xmin=xmin, xmax=xmax, ymin=xmin, ymax=xmax, transform=transform2d)
	progressbar_2d = make_progressbar_2d(s2)
	plot3tensorcomponents(t, dim, pdim; timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=false, progressbar=progressbar_2d, hsize=12Compose.inch, vsize=6Compose.inch, order=order[filter], kw...)
end

function plotall3tensorsandcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; mask=nothing, csize::Tuple=TensorToolbox.mrank(t.core), transpose=false, xtitle="Time", ytitle="Magnitude", timescale::Bool=true, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", functionname="mean", order=gettensorcomponentorder(t, dim; method=:factormagnitude), xmin=datestart, xmax=dateend, ymin=nothing, ymax=nothing, prefix=nothing, maxcomponent=true, savetensorslices=false, transform=nothing, transform2d=nothing, kw...)
	ndimensons = length(t.factors)
	if !checkdimension(dim, ndimensons) || !checkdimension(pdim, ndimensons)
		return
	end
	nc = size(t.factors[pdim], 2)
	np = convert(Int, ceil(nc / 3))
	x = reshape(collect(1:3*np), (3, np))
	x[x.>nc] .= nc
	X = gettensorcomponents(t, dim, pdim; transpose=transpose, csize=csize, prefix=prefix, mask=mask, transform=transform, order=order, maxcomponent=maxcomponent, savetensorslices=savetensorslices)
	for i = 1:np
		filter = vec(x[:,i])
		s2 = plot2dtensorcomponents(t, dim; xtitle=xtitle, ytitle=ytitle, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=true, code=true, order=order, filter=filter, xmin=xmin, xmax=xmax, ymin=xmin, ymax=xmax, transform=transform2d)
		progressbar_2d = make_progressbar_2d(s2)
		prefixnew = prefix == "" ? "" : prefix * "-$(join(filter, "_"))"
		plot3tensorcomponents(t, dim, pdim; csize=csize, transpose=transpose, timescale=timescale, datestart=datestart, dateend=dateend, dateincrement=dateincrement, quiet=false, progressbar=progressbar_2d, hsize=12Compose.inch, vsize=6Compose.inch, order=order[filter], prefix=prefixnew, X=X, maxcomponent=maxcomponent, savetensorslices=savetensorslices, mask=mask, kw...)
	end
end

function plot3maxtensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; kw...)
	plot3tensorcomponents(t, dim, pdim; kw..., maxcomponent=true)
end

function plot3tensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), prefix::String="", filter=(), mask=nothing, transform=nothing, order=gettensorcomponentorder(t, dim; method=:factormagnitude), maxcomponent::Bool=false, savetensorslices::Bool=false, X=nothing, kw...)
	if X == nothing
		X = gettensorcomponents(t, dim, pdim; transpose=transpose, csize=csize, prefix=prefix, filter=filter, mask=mask, transform=transform, order=order, maxcomponent=maxcomponent, savetensorslices=savetensorslices)
	end
	pt = getptdimensions(pdim, length(csize), transpose)
	barratio = (maxcomponent) ? 1/2 : 1/3
	plot3tensors(permutedims(X[order[1]], pt), permutedims(X[order[2]], pt), permutedims(X[order[3]], pt), 1; prefix=prefix, barratio=barratio, kw...)
	if maxcomponent && prefix != ""
		recursivemkdir(prefix)
		mv("$prefix-frame000001.png", "$prefix-max.png"; remove_destination=true)
	end
end

function plotalltensorcomponents(t::TensorDecompositions.Tucker, dim::Integer=1, pdim::Integer=dim; transpose::Bool=false, csize::Tuple=TensorToolbox.mrank(t.core), prefix::String="", filter=(), mask=nothing, transform=nothing, order=gettensorcomponentorder(t, dim; method=:factormagnitude), savetensorslices::Bool=false, quiet=false, kw...)
	X = gettensorcomponents(t, dim, pdim; transpose=transpose, csize=csize, prefix=prefix, filter=filter, mask=mask, transform=transform, order=order, maxcomponent=true, savetensorslices=savetensorslices)
	pt = getptdimensions(pdim, length(csize), transpose)
	mdfilter = ntuple(k->(k == 1 ? 1 : Colon()), length(csize))
	for i = 1:length(X)
		filename = prefix == "" ? "" : "$prefix-tensorslice$i.png"
		p = plotmatrix(permutedims(X[order[i]], pt)[mdfilter...]; filename=filename, kw...)
		!quiet && (info("Slice $i"); display(p); println();)
	end
end

function plot2matrices(X1::Matrix, X2::Matrix; kw...)
	plot2tensors([X1], [X2], 1; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), kw...)
end

function plot2tensors(X1::Array, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; kw...)
	X2 = TensorDecompositions.compose(T2)
	plot2tensors(X1, X2, dim; kw...)
end

function plot2tensors(X1::AbstractArray{T,N}, X2::AbstractArray{T,N}, dim::Integer=1; mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), minvalue2=minvalue, maxvalue2=maxvalue, movie::Bool=false, hsize=12Compose.inch, vsize=6Compose.inch, title::String="", moviedir::String=".", prefix::String = "", keyword="frame", ltitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateend=(datestart != nothing) ? datestart + eval(parse(dateincrement))(sizes[dim]) : nothing, dateincrement::String="Dates.Day", progressbar=progressbar_regular, uniformscaling::Bool=true, colormap=colormap_gyr, vspeed=1.0, kw...) where {T,N}
	if !checkdimension(dim, N)
		return
	end
	recursivemkdir(prefix)
	if !uniformscaling
		minvalue = minimumnan(X1)
		maxvalue = maximumnan(X1)
		minvalue2 = minimumnan(X2)
		maxvalue2 = maximumnan(X2)
	end
	recursivemkdir(moviedir; filename=false)
	@assert sizes == size(X2)
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		g1 = plotmatrix(X1[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=ltitle, colormap=colormap, kw...)
		g2 = plotmatrix(X2[nt...]; minvalue=minvalue2, maxvalue=maxvalue2, title=rtitle, colormap=colormap, kw...)
		if title != ""
			t = Compose.compose(Compose.context(0, 0, 1Compose.w, 0.0001Compose.h),
				(Compose.context(), Compose.fill("gray"), Compose.fontsize(20Compose.pt), Compose.text(0.5Compose.w, 0, title * " : " * sprintf("%06d", i), Compose.hcenter, Compose.vtop)))
		else
			t = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		if progressbar != nothing
			f = progressbar(i, timescale, timestep, datestart, dateend, dateincrement)
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		!quiet && (sizes[dim] > 1) && println(framename)
		!quiet && (Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(t, Compose.hstack(g1, g2), f)); println())
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Compose.vstack(t, Compose.hstack(g1, g2), f))
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-$(keyword)%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -filter:v "setpts=$vspeed*PTS" -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		if moviedir == "."
			moviedir, prefix = splitdir(prefix)
			if moviedir == ""
				moviedir = "."
			end
		end
		cleanup && run(`find $moviedir -name $prefix-$(keyword)"*.png" -delete`)
	end
end

function plot3matrices(X1::Matrix, X2::Matrix, X3::Matrix; kw...)
	plot3tensors([X1], [X2], [X3], 1; minvalue=minimumnan([X1 X2 X3]), maxvalue=maximumnan([X1 X2 X3]), kw...)
end

function plotcmptensors(X1::Array, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; center=true, transform=nothing, mask=nothing, kw...)
	X2 = TensorDecompositions.compose(T2)
	if transform != nothing
		X2 = transform.(X2)
	end
	nanmask!(X2, mask)
	plot2tensors(X1, X2, dim; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), kw...)
end

function plot3tensors(X1::AbstractArray{T,N}, X2::AbstractArray{T,N}, X3::AbstractArray{T,N}, dim::Integer=1; mdfilter=ntuple(k->(k == dim ? dim : Colon()), N), minvalue=minimumnan([X1 X2 X3]), maxvalue=maximumnan([X1 X2 X3]), minvalue2=minvalue, maxvalue2=maxvalue, minvalue3=minvalue, maxvalue3=maxvalue, prefix::String="", keyword="frame", movie::Bool=false, hsize=24Compose.inch, vsize=6Compose.inch, moviedir::String=".", ltitle::String="", ctitle::String="", rtitle::String="", quiet::Bool=false, cleanup::Bool=true, sizes=size(X1), timescale::Bool=true, timestep=1/sizes[dim], datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day", progressbar=progressbar_regular, barratio::Number=1/2, colormap=colormap_gyr, uniformscaling::Bool=true, vspeed=1.0, kw...) where {T,N}
	recursivemkdir(prefix)
	if !checkdimension(dim, N)
		return
	end
	if !uniformscaling
		minvalue = minimumnan(X1)
		maxvalue = maximumnan(X1)
		minvalue2 = minimumnan(X2)
		maxvalue2 = maximumnan(X2)
		minvalue3 = minimumnan(X3)
		maxvalue3 = maximumnan(X3)
	end
	recursivemkdir(moviedir; filename=false)
	@assert sizes == size(X2)
	@assert sizes == size(X3)
	dimname = namedimension(N; char="D", names=("Row", "Column", "Layer"))
	for i = 1:sizes[dim]
		framename = "$(dimname[dim]) $i / $(sizes[dim])"
		nt = ntuple(k->(k == dim ? i : mdfilter[k]), N)
		g1 = plotmatrix(X1[nt...]; minvalue=minvalue, maxvalue=maxvalue, title=ltitle, colormap=colormap, kw...)
		g2 = plotmatrix(X2[nt...]; minvalue=minvalue2, maxvalue=maxvalue2, title=ctitle, colormap=colormap, kw...)
		g3 = plotmatrix(X3[nt...]; minvalue=minvalue3, maxvalue=maxvalue3, title=rtitle, colormap=colormap, kw...)
		if progressbar != nothing
			if sizes[dim] == 1
				f = progressbar(0, timescale, timestep, datestart, dateend, dateincrement)
			else
				f = progressbar(i, timescale, timestep, datestart, dateend, dateincrement)
			end
		else
			f = Compose.compose(Compose.context(0, 0, 1Compose.w, 0Compose.h))
		end
		if !quiet
			(sizes[dim] > 1) && println(framename)
			if typeof(f) != Compose.Context
				Gadfly.draw(Gadfly.PNG(hsize, vsize), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), Compose.hstack(g1, g2, g3)), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f)))); println()
			else
				Gadfly.draw(Gadfly.PNG(hsize, vsize), Compose.vstack(Compose.hstack(g1, g2, g3), f)); println()
			end
		end
		if prefix != ""
			filename = setnewfilename(prefix, i; keyword=keyword)
			if typeof(f) != Compose.Context
				Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Gadfly.vstack(Compose.compose(Compose.context(0, 0, 1, 1 - barratio), Compose.hstack(g1, g2, g3)), Compose.compose(Compose.context(0, 0, 1, barratio), Gadfly.render(f))))
			else
				Gadfly.draw(Gadfly.PNG(joinpath(moviedir, filename), hsize, vsize, dpi=150), Compose.vstack(Compose.hstack(g1, g2, g3), f))
			end
		end
	end
	if movie && prefix != ""
		c = `ffmpeg -i $moviedir/$prefix-$(keyword)%06d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -filter:v "setpts=$vspeed*PTS" -y $moviedir/$prefix.mp4`
		if quiet
			run(pipeline(c, stdout=DevNull, stderr=DevNull))
		else
			run(c)
		end
		if moviedir == "."
			moviedir, prefix = splitdir(prefix)
			if moviedir == ""
				moviedir = "."
			end
		end
		cleanup && run(`find $moviedir -name $prefix-$(keyword)"*.png" -delete`)
	end
end

function plotleftmatrix(X1::Matrix, X2::Matrix; kw...)
	plot3tensors([X1], [X2], [X2.-X1], 1; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), kw...)
end

function plotlefttensor(X1::Array, X2::Array, dim::Integer=1; minvalue=minimumnan([X1 X2]), maxvalue=maximumnan([X1 X2]), minvalue3=nothing, maxvalue3=nothing, center=true, kw...)
	D = X2 - X1
	minvalue3 = minvalue3 == nothing ? minimumnan(D) : minvalue3
	maxvalue3 = maxvalue3 == nothing ? maximumnan(D) : maxvalue3
	if center
		minvalue3, maxvalue3 = min(minvalue3, -maxvalue3), max(maxvalue3, -minvalue3)
	end
	plot3tensors(X1, X2, D, dim; minvalue=minvalue, maxvalue=maxvalue, minvalue3=minvalue3, maxvalue3=maxvalue3, kw...)
end

function plotlefttensor(X1::Array, T2::Union{TensorDecompositions.Tucker,TensorDecompositions.CANDECOMP}, dim::Integer=1; minvalue=nothing, maxvalue=nothing, minvalue3=nothing, maxvalue3=nothing, center=true, transform=nothing, mask=nothing, kw...)
	X2 = TensorDecompositions.compose(T2)
	if transform != nothing
		X2 = transform.(X2)
	end
	D = X2 - X1
	nanmask!(X2, mask)
	nanmask!(D, mask)
	minvalue = minvalue == nothing ? minimumnan([X1 X2]) : minvalue
	maxvalue = minvalue == nothing ? maximumnan([X1 X2]) : maxvalue
	minvalue3 = minvalue3 == nothing ? minimumnan(D) : minvalue3
	maxvalue3 = maxvalue3 == nothing ? maximumnan(D) : maxvalue3
	if center
		minvalue3, maxvalue3 = min(minvalue3, -maxvalue3), max(maxvalue3, -minvalue3)
	end
	plot3tensors(X1, X2, D, dim; minvalue=minvalue, maxvalue=maxvalue, minvalue3=minvalue3, maxvalue3=maxvalue3, kw...)
end

function setnewfilename(filename::String, frame::Integer=0; keyword::String="frame")
	dir = dirname(filename)
	fn = splitdir(filename)[end]
	fs = split(fn, ".")
	if length(fs) == 1
		root = fs[1]
		ext = ""
	else
		root = join(fs[1:end-1], ".")
		ext = fs[end]
	end
	if ext == ""
		ext = "png"
		fn = fn * "." * ext
	end
	if !contains(fn, keyword)
		fn = root * "-$(keyword)000000." * ext
	end
	if VERSION >= v"0.7"
		rtest = occursin(Regex(string("-", keyword, "[0-9]*[.].*\$")), fn)
	else
		rtest = ismatch(Regex(string("-", keyword, "[0-9]*[.].*\$")), fn)
	end
	if rtest
		rm = match(Regex(string("-", keyword, "([0-9]*)[.](.*)\$")), fn)
		if frame == 0
			v = parse(Int, rm.captures[1]) + 1
		else
			v = frame

		end
		l = length(rm.captures[1])
		f = "%0" * string(l) * "d"
		filename = "$(fn[1:rm.offset-1])-$(keyword)$(sprintf(f, v)).$(rm.captures[2])"
		return joinpath(dir, filename)
	else
		warn("setnewfilename failed!")
		return ""
	end
end

"""
colors=[parse(Colors.Colorant, "green"), parse(Colors.Colorant, "orange"), parse(Colors.Colorant, "blue"), parse(Colors.Colorant, "gray")]
gm=[Gadfly.Guide.manual_color_key("", ["Oil", "Gas", "Water"], colors[1:3]), Gadfly.Theme(major_label_font_size=16Gadfly.pt, key_label_font_size=14Gadfly.pt, minor_label_font_size=12Gadfly.pt)]
"""
function plot2d(T::Array, Te::Array=T; quiet::Bool=false, wellnames=nothing, Tmax=nothing, Tmin=nothing, xtitle::String="", ytitle::String="", titletext::String="", figuredir::String="results", hsize=8Gadfly.inch, vsize=4Gadfly.inch, keyword::String="", dimname::String="Column", colors=NTFk.colors, gm=[Gadfly.Theme(major_label_font_size=16Gadfly.pt, key_label_font_size=14Gadfly.pt, minor_label_font_size=12Gadfly.pt)], linewidth::Measures.Length{:mm,Float64}=2Gadfly.pt, xaxis=1:size(Te,2), xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, xintercept=[])
	recursivemkdir(figuredir)
	c = size(T)
	if length(c) == 2
		nlayers = 1
	else
		nlayers = c[3]
	end
	if wellnames != nothing
		@assert length(wellnames) == c[1]
	end
	@assert c == size(Te)
	@assert length(vec(collect(xaxis))) == c[2]
	if Tmax != nothing && Tmin != nothing
		@assert size(Tmax) == size(Tmin)
		@assert size(Tmax, 1) == c[1]
		@assert size(Tmax, 2) == c[3]
		append = ""
	else
		if maximum(T) <= 1. && maximum(Te) <= 1.
			append = "_normalized"
		else
			append = ""
		end
	end
	if keyword != ""
		append *= "_$(keyword)"
	end
	for w = 1:c[1]
		!quiet && (if wellnames != nothing
			println("$dimname $w : $(wellnames[w])")
		else
			println("$dimname $w")
		end)
		p = Vector{Any}(nlayers * 2)
		pc = 1
		for i = 1:nlayers
			if nlayers == 1
				y = T[w,:]
				ye = Te[w,:]
			else
				y = T[w,:,i]
				ye = Te[w,:,i]
			end
			if Tmax != nothing && Tmin != nothing
				y = y * (Tmax[w,i] - Tmin[w,i]) + Tmin[w,i]
				ye = ye * (Tmax[w,i] - Tmin[w,i]) + Tmin[w,i]
			end
			p[pc] = Gadfly.layer(x=xaxis, y=y, xintercept=xintercept, Gadfly.Geom.line, Gadfly.Theme(line_width=linewidth, default_color=colors[i]), Gadfly.Geom.vline)
			pc += 1
			p[pc] = Gadfly.layer(x=xaxis, y=ye, xintercept=xintercept, Gadfly.Geom.line, Gadfly.Theme(line_style=:dot, line_width=linewidth, default_color=colors[i]), Gadfly.Geom.vline)
			pc += 1
		end
		if wellnames != nothing
			tm = [Gadfly.Guide.title("$dimname $(wellnames[w]) $titletext")]
			if dimname != ""
				filename = "$(figuredir)/$(lowercase(dimname))_$(wellnames[w])$(append).png"
			else
				filename = "$(figuredir)/$(wellnames[w])$(append).png"
			end
		else
			tm = []
			if dimname != ""
				filename = "$(figuredir)/$(lowercase(dimname))$(append).png"
			else
				filename = "$(figuredir)/$(append[2:end]).png"
			end
		end
		yming = ymin
		ymaxg = ymax
		if ymin != nothing && length(ymin) > 1
			yming = ymin[w]
		end
		if ymax != nothing && length(ymax) > 1
			ymaxg = ymax[w]
		end
		f = Gadfly.plot(p..., tm..., Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=yming, ymax=ymaxg))
		Gadfly.draw(Gadfly.PNG(filename, hsize, vsize, dpi=300), f)
		!quiet && (display(f); println())
	end
end

function progressbar_regular(i::Number, timescale::Bool=false, timestep::Number=1, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day")
	s = timescale ? sprintf("%6.4f", i * timestep) : sprintf("%6d", i)
	if datestart != nothing
		if dateend != nothing
			s = datestart + ((dateend .- datestart) * (i-1) * timestep)
		else
			s = datestart + eval(parse(dateincrement))(i-1)
		end
	end
	return Compose.compose(Compose.context(0, 0, 1Compose.w, 0.05Compose.h),
		(Compose.context(), Compose.fill("gray"), Compose.fontsize(10Compose.pt), Compose.text(0.01, 0.0, s, Compose.hleft, Compose.vtop)),
		(Compose.context(), Compose.fill("tomato"), Compose.rectangle(0.75, 0.0, i * timestep * 0.2, 5)),
		(Compose.context(), Compose.fill("gray"), Compose.rectangle(0.75, 0.0, 0.2, 5)))
end

function make_progressbar_2d(s)
	function progressbar_2d(i::Number, timescale::Bool=false, timestep::Number=1, datestart=nothing, dateend=nothing, dateincrement::String="Dates.Day")
		if i > 0
			xi = timescale ? i * timestep : i
			if datestart != nothing
				if dateend != nothing
					xi = datestart + ((dateend .- datestart) * (i-1) * timestep)
				else
					xi = datestart + eval(parse(dateincrement))(i-1)
				end
			end
			return Gadfly.plot(s..., Gadfly.layer(xintercept=[xi], Gadfly.Geom.vline(color=["gray"], size=[2Gadfly.pt])))
		else
			return Gadfly.plot(s...)
		end
	end
	return progressbar_2d
end