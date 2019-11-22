# local build environment

function travis_before_install {
    sudo apt-get update;
    sudo apt-get install flawfinder squashfs-tools uuid-dev libuuid1 libffi-dev libssl-dev libssl1.0.0 \
    libarchive-dev libgpgme11-dev libseccomp-dev wget gcc make pkg-config -y;
    export VERSION=3.5.0;
    travis_retry wget -q https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz;
    tar -xzf singularity-${VERSION}.tar.gz;
    cd singularity;
    ./mconfig;
    make -C ./builddir;
    sudo make -C ./builddir install;
    cd -;
    travis_retry wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh;
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda;
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
}

function travis_install {
    python setup.py develop
    # Verify import with bare install
    python -c 'import pydra; print(pydra.__version__)'
}

function travis_before_script {
    pip install -e ".[test]"
}

function travis_script {
    pytest -vs --cov pydra --cov-config .coveragerc --cov-report xml:cov.xml --doctest-modules pydra
}

function travis_after_script {
    codecov --file cov.xml --flags unittests -e TRAVIS_JOB_NUMBER
}
