if [ -z "$1" ]; then
    TEST="pydra"
else
    TEST=$1
fi


docker exec pydra-slurm-docker bash -c "pytest -vv -s /pydra/$TEST --color=yes -vs -k 'not test_audit_prov and not test_audit_prov_messdir_1 and not test_audit_prov_messdir_2 and not test_audit_prov_wf and not test_audit_all' $2"
