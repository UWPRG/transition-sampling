# CP2K v7.1.0, GROMACS 2020.6, and PLUMED v2.6.2
FROM ubuntu:latest AS base

SHELL ["/bin/bash", "-c"]
# Libaries that are linked to, needed for building and running
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                libblas-dev \
                liblapack-dev \
                libopenblas-dev \
                pkg-config \
		        git \
		        python3.8 \
                python3-pip \
		        libfftw3-dev \
		        libgsl0-dev \
                vim-tiny \
		        libz-dev && apt-get clean && \
                            rm -r /var/lib/apt && \
                            ln -s /usr/bin/python3.8 /usr/bin/python
# Link python3.8 as default
WORKDIR /src

FROM base as build
# Tools only needed for building
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential \
    	        gcc \
		        g++ \
		        gfortran \
		        wget \
		        cmake \
                xxd \
		        &&  apt-get clean && \
                    rm -r /var/lib/apt
RUN git clone --depth 1 --branch v7.1.0 https://github.com/cp2k/cp2k.git /src/cp2k
RUN git clone --depth 1 --branch v2.6.2 https://github.com/plumed/plumed2.git /src/plumed

# Build PLUMED
WORKDIR /src/plumed
RUN ./configure && make -j 16

# Build CP2K
WORKDIR /src/cp2k
RUN git checkout v7.1.0

# Need to pull down DBCSR submodule
RUN git submodule update --init --recursive

# Link libraries so ld can find them
RUN ln -s /src/plumed/src/lib/libplumed.so /usr/local/lib/libplumed.so
RUN ln -s /src/plumed/src/lib/libplumedKernel.so /usr/local/lib/libplumedKernel.so

# Run the setup, classical only, no mpi
RUN cd tools/toolchain && \
    ./install_cp2k_toolchain.sh --with-fftw=system --with-reflapack=no --with-cmake=system \
         --mpi-mode=no --with-openblas=system --with-libint=no --with-libxc=no --with-gsl=system  \
         --with-hdf5=no --with-spglib=install --with-plumed=system && \
    rm -rf build/
RUN cp tools/toolchain/install/arch/* arch

# Update arch to include plumed
RUN echo "include /src/plumed/src/lib/Plumed.inc.shared" >> arch/local.ssmp
RUN echo "EXTERNAL_OBJECTS=$(PLUMED_STATIC_DEPENDENCIES)" >> arch/local.ssmp
RUN sed -i 's/LIBS\s\+=\(.*\)/LIBS=\1 -lz -ldl -lstdc++ -lplumed/' arch/local.ssmp
RUN sed -i 's/DFLAGS\s\+=\(.*\)/DFLAGS=\1 -D__PLUMED2/' arch/local.ssmp 

RUN source tools/toolchain/install/setup
RUN make -j 16 ARCH=local VERSION=ssmp
RUN make -j 16 ARCH=local VERSION=ssmp clean


# build GROMACS
RUN git clone --depth 1 --branch v2020.6 https://github.com/gromacs/gromacs /src/gromacs

# option 4 in patch is gromacs2020.4
RUN cd /src/gromacs/ && \
    source /src/plumed/sourceme.sh && \
    echo 4 | plumed patch -p && \
    mkdir build && \
    cd /src/gromacs/build && \
    cmake .. && make -j 16 install && make clean
RUN ldconfig


# runner is what we will actually run on. Copy all the built binaries that are
# linked to libraries in base.
FROM base AS runner
RUN mkdir -p /src/plumed/src/lib && mkdir -p /src/cp2k/tools/toolchain/ \
    && mkdir -p /usr/local/gromacs/
COPY --from=build /src/plumed/src/lib/ /src/plumed/src/lib/
COPY --from=build /src/plumed/sourceme.sh /src/plumed/
COPY --from=build /src/cp2k/exe/local/cp2k.ssmp /src/
COPY --from=build /src/cp2k/tools/toolchain/install /src/cp2k/tools/toolchain/install/

COPY --from=build /usr/local/gromacs/ /usr/local/gromacs/
RUN ln -s /usr/local/gromacs/bin/gmx /usr/bin/gmx
