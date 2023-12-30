FROM centos:7.9.2009

LABEL org.opencontainers.image.source="https://github.com/giovtorres/docker-centos7-slurm" \
      org.opencontainers.image.title="docker-centos7-slurm" \
      org.opencontainers.image.description="Slurm All-in-one Docker container on CentOS 7" \
      org.label-schema.docker.cmd="docker run -it -h slurmctl giovtorres/docker-centos7-slurm:latest" \
      maintainer="Giovanni Torres"

ENV PATH "/root/.pyenv/shims:/root/.pyenv/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin"

EXPOSE 6817 6818 6819 6820 3306

# Install common YUM dependency packages
# The IUS repo install epel-release as a dependency while also providing a newer version of Git
RUN set -ex \
    && yum makecache fast \
    && yum -y update \
    && yum -y install https://repo.ius.io/ius-release-el7.rpm \
    && yum -y install \
        autoconf \
        bash-completion \
        bzip2 \
        bzip2-devel \
        file \
        iproute \
        gcc \
        gcc-c++ \
        gdbm-devel \
        git236 \
        glibc-devel \
        gmp-devel \
        libffi-devel \
        libGL-devel \
        libX11-devel \
        make \
        mariadb-server \
        mariadb-devel \
        munge \
        munge-devel \
        ncurses-devel \
        patch \
        perl-core \
        pkgconfig \
        psmisc \
        readline-devel \
        sqlite-devel \
        tcl-devel \
        tix-devel \
        tk \
        tk-devel \
        supervisor \
        wget \
        which \
        vim-enhanced \
        xz-devel \
        zlib-devel http-parser-devel json-c-devel libjwt-devel libyaml-devel \
    && yum clean all \
    && rm -rf /var/cache/yum

# Set Vim and Git defaults
RUN set -ex \
    && echo "syntax on"           >> "$HOME/.vimrc" \
    && echo "set tabstop=4"       >> "$HOME/.vimrc" \
    && echo "set softtabstop=4"   >> "$HOME/.vimrc" \
    && echo "set shiftwidth=4"    >> "$HOME/.vimrc" \
    && echo "set expandtab"       >> "$HOME/.vimrc" \
    && echo "set autoindent"      >> "$HOME/.vimrc" \
    && echo "set fileformat=unix" >> "$HOME/.vimrc" \
    && echo "set encoding=utf-8"  >> "$HOME/.vimrc" \
    && git config --global color.ui auto \
    && git config --global push.default simple

# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini.asc /tini.asc
RUN gpg --batch --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 595E85A6B1B4779EA4DAAEC70B588DFF0527A9B7 \
 && gpg --batch --verify /tini.asc /tini
RUN chmod +x /tini

# Install OpenSSL1.1.1
# See PEP 644: https://www.python.org/dev/peps/pep-0644/
ARG OPENSSL_VERSION="1.1.1s"
RUN set -ex \
    && wget --quiet https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz \
    && tar xzf openssl-${OPENSSL_VERSION}.tar.gz \
    && pushd openssl-${OPENSSL_VERSION} \
    && ./config --prefix=/opt/openssl --openssldir=/etc/ssl \
    && make \
    && make test \
    && make install \
    && echo "/opt/openssl/lib" >> /etc/ld.so.conf.d/openssl.conf \
    && ldconfig \
    && popd \
    && rm -f openssl-${OPENSSL_VERSION}.tar.gz

# Install supported Python versions and install dependencies.
# Set the default global to the latest supported version.
# Use pyenv inside the container to switch between Python versions.
ARG PYTHON_VERSIONS="3.8.16 3.9.16 3.10.9 3.11.1"
ARG CONFIGURE_OPTS="--with-openssl=/opt/openssl"
RUN set -ex \
    && curl https://pyenv.run | bash \
    && echo "eval \"\$(pyenv init --path)\"" >> "${HOME}/.bashrc" \
    && echo "eval \"\$(pyenv init -)\"" >> "${HOME}/.bashrc" \
    && source "${HOME}/.bashrc" \
    && pyenv update \
    && for python_version in ${PYTHON_VERSIONS}; \
        do \
            pyenv install $python_version; \
            pyenv global $python_version; \
            pip install Cython pytest; \
        done

# Compile, build and install Slurm from Git source
ARG SLURM_TAG=slurm-23-02-1-1
ARG JOBS=4
RUN set -ex \
    && git clone -b ${SLURM_TAG} --single-branch --depth=1 https://github.com/SchedMD/slurm.git \
    && pushd slurm \
    && ./configure --prefix=/usr --sysconfdir=/etc/slurm --enable-slurmrestd \
        --with-mysql_config=/usr/bin --libdir=/usr/lib64 --enable-multiple-slurmd \
    && sed -e 's|#!/usr/bin/env python3|#!/usr/bin/python|' -i doc/html/shtml2html.py \
    && make -j ${JOBS} install \
    && install -D -m644 etc/cgroup.conf.example /etc/slurm/cgroup.conf.example \
    && install -D -m644 etc/slurm.conf.example /etc/slurm/slurm.conf.example \
    && install -D -m600 etc/slurmdbd.conf.example /etc/slurm/slurmdbd.conf.example \
    && install -D -m644 contribs/slurm_completion_help/slurm_completion.sh /etc/profile.d/slurm_completion.sh \
    && popd \
    && rm -rf slurm \
    && groupadd -r slurm  \
    && useradd -r -g slurm slurm \
    && mkdir -p /etc/sysconfig/slurm \
        /var/spool/slurmd \
        /var/spool/slurmctld \
        /var/log/slurm \
        /var/run/slurm \
    && chown -R slurm:slurm /var/spool/slurmd \
        /var/spool/slurmctld \
        /var/log/slurm \
        /var/run/slurm \
    && /sbin/create-munge-key

RUN dd if=/dev/random of=/etc/slurm/jwt_hs256.key bs=32 count=1 \
    && chmod 600 /etc/slurm/jwt_hs256.key && chown slurm:slurm /etc/slurm/jwt_hs256.key

COPY --chown=slurm files/slurm/slurm.conf files/slurm/slurmdbd.conf files/slurm/cgroup.conf /etc/slurm/
COPY files/supervisord.conf /etc/

RUN chmod 0600 /etc/slurm/slurmdbd.conf

# Mark externally mounted volumes
VOLUME ["/var/lib/mysql", "/var/lib/slurmd", "/var/spool/slurm", "/var/log/slurm"]

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
