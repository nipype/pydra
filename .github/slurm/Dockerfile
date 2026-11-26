# Single-container Slurm for the test-slurm CI job: slurmctld, slurmd, slurmdbd (MariaDB) and munge.
# Slurm is built from the pinned upstream tarball; Python is installed by the job, not here.
# ubuntu:24.04, pinned by manifest-list digest
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852

ARG SLURM_VERSION=25.11.8
ARG SLURM_SHA256=34ace13f81011add6094569d13bfc4006ad8868201c2236e2905443c7e526393

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential bzip2 ca-certificates curl git gosu libmariadb-dev libmunge-dev libssl-dev \
        mariadb-server munge pkg-config procps python3-minimal \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLO https://download.schedmd.com/slurm/slurm-${SLURM_VERSION}.tar.bz2 \
    && echo "${SLURM_SHA256}  slurm-${SLURM_VERSION}.tar.bz2" | sha256sum -c - \
    && tar xjf slurm-${SLURM_VERSION}.tar.bz2 \
    && cd slurm-${SLURM_VERSION} \
    && ./configure --prefix=/usr --sysconfdir=/etc/slurm \
    && make -j"$(nproc)" install \
    && cd .. && rm -rf slurm-${SLURM_VERSION} slurm-${SLURM_VERSION}.tar.bz2 \
    && apt-get purge -y --auto-remove build-essential libmariadb-dev libmunge-dev libssl-dev pkg-config \
    && apt-get install -y --no-install-recommends libmariadb3 libmunge2 libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r slurm && useradd -r -g slurm slurm \
    && mkdir -p /etc/slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm /run/slurm \
    && chown -R slurm:slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm /run/slurm \
    && (dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key 2>/dev/null) \
    && chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key

RUN git config --system --add safe.directory "*"

COPY slurmdbd.conf /etc/slurm/slurmdbd.conf
COPY entrypoint.sh /usr/local/bin/slurm-entrypoint.sh
RUN chown slurm:slurm /etc/slurm/slurmdbd.conf && chmod 600 /etc/slurm/slurmdbd.conf \
    && chmod +x /usr/local/bin/slurm-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/slurm-entrypoint.sh"]
