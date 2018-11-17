import Gadfly

function imshow(x, title::String="", units::String="", args...)
	is, js, values = findnz(x)
	m, n = size(x)
	df = DataFrames.DataFrame(i=is, j=js, value=values)
	Gadfly.plot(df, x="j", y="i", color="value", xmin=[1,1], ymin=[1,1], xmax=[1,n], ymax=[1,m],
				Gadfly.Coord.cartesian(yflip=true, fixed=true),
				Gadfly.Scale.x_continuous(minvalue=0.5, maxvalue=n+0.5),
				Gadfly.Scale.y_continuous(minvalue=0.5, maxvalue=m+0.5),
				Gadfly.Geom.rectbin,
				Gadfly.Guide.title(title), Gadfly.Guide.colorkey(title=units),
				Gadfly.Guide.XLabel(""), Gadfly.Guide.YLabel(""),
			 	Gadfly.Guide.xticks(ticks=[1,n]), Gadfly.Guide.yticks(ticks=[1,m]),
				Gadfly.Theme(grid_color="black", grid_line_width=0Gadfly.pt),
				args...)
end

plot(Gadfly.layer(x=rand(10)*10, y=rand(10)*10, Gadfly.Geom.line()), layer(x=js, y=is, color=values, Geom.rectbin), Scale.x_continuous, Scale.y_continuous, Coord.cartesian(yflip=true, fixed=true, xmin=0.5, xmax=m+.5, ymin=0.5, ymax=n+.5))