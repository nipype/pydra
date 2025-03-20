PKG_DIR=$(realpath $(dirname $0)/..)
NO_ET=TRUE

# Pull image
docker build -t pydra-slurm-docker $PKG_DIR/docker-scripts-for-tests

# Start image
docker run --rm -itd -h slurmctl --cap-add sys_admin -d --name pydra-slurm-docker -v $PKG_DIR:/pydra -e NO_ET=$NO_ET pydra-slurm-docker

# Display previous jobs with sacct
echo "Allowing ports/daemons time to start" && sleep 20
docker exec pydra-slurm-docker bash -c "sacctmgr -i add account none,test Cluster=linux Description='none' Organization='none'"
docker exec pydra-slurm-docker bash -c "sacct && sinfo && squeue" 2&> /dev/null
if [ $? -ne 0 ]; then
    echo "Slurm docker image error"
    exit 1
fi

# Setup Python
docker exec pydra-slurm-docker bash -c "echo $NO_ET"
docker exec pydra-slurm-docker bash -c "ls -la && echo list top level dir"
docker exec pydra-slurm-docker bash -c "ls -la /pydra && echo list pydra dir"
docker exec pydra-slurm-docker bash -c "pip install --upgrade pip && pip install -e /pydra[test,psij] && python -c 'import pydra.engine; print(pydra.engine.__version__)'"
