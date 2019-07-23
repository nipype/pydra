# docker environment for slurm worker testing

export DOCKER_IMAGE="mgxd/slurm:19.05.1"

function travis_before_install {
    docker pull ${DOCKER_IMAGE}
    # have image running in background
    export CONTAINER_ID=$(docker run -itd -h ernie -v `pwd`:/pydra ${DOCKERIMAGE})
    docker exec ${CONTAINER_ID} bash -c "sacct && sinfo && squeue" 2&> /dev/null
    if [ $? -ne 0 ]; then
        echo "Slurm docker image error"
        exit 1
    fi
}

function travis_install {
    docker exec ${CONTAINER_ID} bash -c "cd /pydra && pip install -e .[test] && python -c 'import pydra; print(pydra.__version__)'"
}

function travis_before_script {
    :
}

function travis_script {
    docker exec ${CONTAINER_ID} bash -c "pytest -vs -n auto --cov pydra --cov-config .coveragerc --cov-report xml:cov.xml --doctest-modules pydra/engine/"
}

function travis_after_script {
    docker exec ${CONTAINER_ID} bash -c "codecov --file cov.xml --flags unittests"
    docker stop ${CONTAINER_ID}
    unset ${CONTAINER_ID}
}