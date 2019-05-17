using Plots
using DelimitedFiles

const serial = readdlm("out-serial.txt")
const darray = readdlm("out-darray.txt")
const mpi = readdlm("out-mpi.txt")
const elemental = readdlm("out-elemental.txt")

# column indices
const procs = 1
const threads = 2
const ddim = 3
const mintime = 4
const maxtime = 5

function filter(col,val,data)
  matching_rows = findall(x->x == val, data[:,col])
  result = similar(data, length(matching_rows), size(data,2))
  for (i,j) in enumerate(matching_rows)
    result[i,:] .= data[j,:]
  end
  return result
end

function plotcols(data, xcol, ycol; kwargs...)
  is = sortperm(data[:,xcol])
  # sorted = sort(data[:,[xcol,ycol]], dims=1)
  # display(sorted)
  x = data[is,xcol]
  y = data[is,ycol]
  plot!(log2.(x), y*1000; xticks=(log2.(x), string.(Int.(x))), kwargs...)
  return (x,y)
end

function plot_comparison(darray_filt, mpi_filt, serial_filt_time, filename, legpos)
  #plotcols(filter(ddim, 1, darray_filt), procs, mintime, label="DArray, rows", marker=:square)
  plotcols(filter(ddim, 2, darray_filt), procs, mintime, label="DistributedArray", color="red", marker=:circle, linewidth=3, markerstrokewidth=0)
  #plotcols(filter(ddim, 1, mpi_filt), procs, mintime, label="MPI, rows", marker=:circle)
  (ideal_x, y) = plotcols(filter(ddim, 2, mpi_filt), procs, mintime, label="MPIArray", color="blue", marker=:circle, linewidth=3, markerstrokewidth=0)
  hline!([serial_filt_time*1000], label="Local.Array", color="orange", linestyle=:dash, linewidth=3)
  plot!(log2.(ideal_x), serial_filt_time*1000 ./ ideal_x, label="Ideal scaling", color="white", linestyle=:dash, linewidth=3, xlabel="Number of processes", ylabel="Time (ms)", legend=legpos, background_color="black", foreground_color="gray", background_color_legend="gray", legendfontcolor="black")
  savefig(filename)
end

plotlyjs()

plot()
plotcols(filter(threads,1,elemental), procs, mintime, label="Elemental", color="green", marker=:circle, linewidth=3, markerstrokewidth=0)
plot_comparison(filter(threads,1,darray), filter(threads,1,mpi), minimum(filter(threads, 1, serial)[:,mintime]), "singlethread.pdf", :top)

plot()
plot_comparison(filter(threads,32,darray), filter(threads,32,mpi), minimum(filter(threads, 32, serial)[:,mintime]), "multithread.pdf", :top)


