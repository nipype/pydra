FROM giovtorres/docker-centos7-slurm:19.05.1

ENV CONDAPATH="~/miniconda"
# install miniconda3
RUN wget -q http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p ${CONDAPATH}
ENV PATH="${CONDAPATH}/bin:${PATH}"
RUN conda update -yq --all conda \
    && conda clean -tipy \
    && conda config --set always_yes yes \
    && conda config --add channels conda-forge \
    && rm ~/miniconda.sh
