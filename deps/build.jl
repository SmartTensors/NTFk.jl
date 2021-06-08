if haskey(ENV, "NTFk_NO_PYTHON")
	@info("No Python will be used ...")
else
	import PyCall

	const PACKAGES = ["matplotlib", "numpy", "tensorly", "pytorch", "tensorflow", "mxnet"]
	@info("Checking for Python packages: $(PACKAGES)...")

	try
		for p in PACKAGES
			@info("Checking python package $(p) ...")
			Core.eval(Main, :(PyCall.pyimport($p)))
		end
		@info("Python packages are already installed!")
	catch errmsg
		println(errmsg)
		@warn("Python packages are missing!")

		try
			@info("Checking for python pip using PyCall ...")
			Core.eval(Main, :(PyCall.pyimport("pip")))
		catch errmsg
			println(errmsg)
			@warn("Python pip is not installed!")
			@info("Downloading & installing python pip ...")
			global get_pip = joinpath(dirname(@__FILE__), "get-pip.py")
			download("https://bootstrap.pypa.io/get-pip.py", get_pip)
			run(`$(PyCall.python) $get_pip --user`)
			rm("$get_pip")
		end

		try
			@info("Installing Python packages using pip ...")
			Core.eval(Main, :(PyCall.pyimport("pip")))
			global proxy_args = String[]
			if haskey(ENV, "http_proxy")
				push!(proxy_args, "--proxy")
				push!(proxy_args, ENV["http_proxy"])
			end
			println("Installing required python packages using pip")
			run(`$(PyCall.python) $(proxy_args) -m pip install --user --upgrade pip setuptools`)
			run(`$(PyCall.python) $(proxy_args) -m pip install --user -c conda`)
			run(`$(PyCall.python) $(proxy_args) -m pip install --user -c anaconda mxnet`)
			run(`$(PyCall.python) $(proxy_args) -m pip install --user -c tensorly tensorly`)
			run(`$(PyCall.python) $(proxy_args) -m pip install --user -c pytorch pytorch`)
			run(`$(PyCall.python) $(proxy_args) -m pip install --user -c pytorch tensorflow`)
			run(`$(PyCall.python) $(proxy_args) -m pip install --user -c anaconda matplotlib`)
		catch errmsg
			println(errmsg)
			@warn("Installing Python packages using pip fails!")
		end

		try
			for p in PACKAGES
				@info("Checking python package $(p) ...")
				Core.eval(Main, :(PyCall.pyimport($p)))
			end
			@info("Python packages are installed using pip!")
		catch errmsg
			println(errmsg)
			@warn("Python package installation using pip has failed!")
			@info("Using Conda instead ...")
			try
			import Conda
				Conda.add("matplotlib")
				Conda.add("numpy")
				Conda.add("tensorly"; channel="tensorly")
				Conda.add("pytorch"; channel="pytorch")
				Conda.add("tensorflow")
				Conda.add("mxnet")
			catch errmsg
				println(errmsg)
				@warn("Installing Python packages using Conda fails!")
			end
		end
	end
end