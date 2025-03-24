if [ -z "$1" ]; then
    TEST="::$1"
else
    TEST=""
fi


docker exec pydra-slurm-docker bash -c "pytest -vv --with-psij --only-slurm -s /pydra/pydra/engine/test_submitter.py$TEST --color=yes -vs $2"
