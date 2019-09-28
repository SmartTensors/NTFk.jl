FROM julia
COPY . /app
COPY . /src

RUN apt -y update

RUN apt -y install git make g++ vim libcairo2 libpango-1.0-0 libpango1.0-0 python3 libpython3.7 python3-pip

RUN pip3 install matplotlib pyyaml

# RUN apt -y gettext-base libcroco3 libglib2.0-0 libglib2.0-data libgpm2 libncurses5 libxml2 sgml-base shared-mime-info xdg-user-dirs xml-core fontconfig-config fonts-dejavu-core libexpat1 libfontconfig1 libfreetype6 libbsd0 libcairo2 libpixman-1-0 libx11-6 libx11-data libxau6 libxcb-render0 libxcb-shm0 libxcb1 libxdmcp6 libxext6 libxrender1 libpng16-16 ucf fontconfig libdatrie1 libgraphite2-3 libharfbuzz0b libpango-1.0-0 libpango1.0-0 libpangocairo-1.0-0 libpangoft2-1.0-0 libpangox-1.0-0 libpangoxft-1.0-0 libthai-data libthai0 libxft2 python3 libpython3.5

RUN julia -e 'import Pkg; ENV["PYTHON"]=""; Pkg.add("PyCall"); Pkg.build("PyCall"); import PyCall'
RUN julia -e 'import Pkg; Pkg.add("Conda"); Pkg.build("Conda"); import Conda; Conda.add("pyyaml"); Conda.add("tensorly"; channel="tensorly"); Conda.add("pytorch"; channel="pytorch"); Conda.add("tensorflow")'
RUN julia -e 'import Pkg; Pkg.add("Revise"); Pkg.build("Revise"); import Revise'
RUN julia -e 'import Pkg; Pkg.develop("Mads"); Pkg.build("Mads"); import Mads'
RUN julia -e 'import Pkg; Pkg.develop("NMFk"); Pkg.build("NMFk"); import NMFk'
RUN julia -e 'import Pkg; Pkg.develop("NTFk"); Pkg.build("NTFk"); import NTFk'

CMD ["julia -e 'import Revise; import NTFk' -i"]
