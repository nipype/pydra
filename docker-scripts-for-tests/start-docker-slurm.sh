PKG_DIR=$(realpath $(dirname $0)/..)
DOCKER_IMAGE=adi611/docker-centos7-slurm:23.02.1
PYTHON_VERSION=3.11.5
NO_ET=TRUE

# Pull image
docker pull $DOCKER_IMAGE

# Start image
docker run -itd -h slurmctl --cap-add sys_admin -d --name slurm -v $PKG_DIR:/pydra -e NO_ET=$NO_ET $DOCKER_IMAGE

# Display previous jobs with sacct
echo "Allowing ports/daemons time to start" && sleep 10
docker exec slurm bash -c "sacctmgr -i add account none,test Cluster=linux Description='none' Organization='none'"
docker exec slurm bash -c "sacct && sinfo && squeue" 2&> /dev/null
if [ $? -ne 0 ]; then
    echo "Slurm docker image error"
    exit 1
fi

# Setup Python
docker exec slurm bash -c "echo $NO_ET"
docker exec slurm bash -c "ls -la && echo list top level dir"
docker exec slurm bash -c "ls -la /pydra && echo list pydra dir"
docker exec slurm bash -c "CONFIGURE_OPTS=\"-with-openssl=/opt/openssl\" pyenv install -v ${PYTHON_VERSION}"
docker exec slurm bash -c "pyenv global ${PYTHON_VERSION}"
docker exec slurm bash -c "pip install --upgrade pip && pip install -e /pydra[test,psij] && python -c 'import pydra.engine; print(pydra.engine.__version__)'"
