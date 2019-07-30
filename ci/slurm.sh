# docker environment for slurm worker testing

export DOCKER_IMAGE="mgxd/slurm:19.05.1"

function travis_before_install {
    export CI_ENV=`bash <(curl -s https://codecov.io/env)`
    docker pull ${DOCKER_IMAGE}
    # have image running in background
    docker run -itd -e CI_ENV=$CI_ENV -h ernie --name slurm -v `pwd`:/pydra ${DOCKER_IMAGE}
    echo "Allowing ports/daemons time to start" && sleep 10
    # ensure sacct displays previous jobs
    # https://github.com/giovtorres/docker-centos7-slurm/issues/3
    docker exec slurm bash -c "sacctmgr -i add cluster name=linux \
        && supervisorctl restart slurmdbd \
        && supervisorctl restart slurmctld \
        && sacctmgr -i add account none,test Cluster=linux Description='none' Organization='none'"
    docker exec slurm bash -c "sacct && sinfo && squeue" 2&> /dev/null
    if [ $? -ne 0 ]; then
        echo "Slurm docker image error"
        exit 1
    fi
}

function travis_install {
    docker exec slurm bash -c "pip install -e /pydra[test] && python -c 'import pydra; print(pydra.__version__)'"
}

function travis_before_script {
    :
}

function travis_script {
    docker exec slurm bash -c "cd /pydra && pytest --color=yes -vs -n auto --cov pydra --cov-config .coveragerc --cov-report xml:cov.xml --doctest-modules pydra"
}

function travis_after_script {
    docker exec slurm bash -c "cd /pydra && codecov --file cov.xml --flags unittests -e $CI_ENV"
    docker stop slurm && docker rm slurm
}
