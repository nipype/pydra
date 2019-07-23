# docker environment for slurm worker testing

export DOCKER_IMAGE="mgxd/slurm:19.05.1"

function travis_before_install {
    docker pull ${DOCKER_IMAGE}
    # have image running in background
    docker run -itd -h ernie --name slurm -v `pwd`:/pydra ${DOCKER_IMAGE}
    echo "Sleeping so ports can wake up" && sleep 10
    docker exec slurm bash -c "sacct && sinfo && squeue" 2&> /dev/null
    if [ $? -ne 0 ]; then
        echo "Slurm docker image error"
        exit 1
    fi
}

function travis_install {
    docker exec slurm bash -c "cd /pydra && pip install -e .[test] && python -c 'import pydra; print(pydra.__version__)'"
}

function travis_before_script {
    :
}

function travis_script {
    docker exec slurm bash -c "cd /pydra && pytest -vs -n auto --cov pydra --cov-config .coveragerc --cov-report xml:cov.xml --doctest-modules pydra"
}

function travis_after_script {
    docker exec slurm bash -c "codecov --file cov.xml --flags unittests"
    docker stop slurm && docker rm slurm
}