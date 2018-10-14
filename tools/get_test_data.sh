#!/bin/bash

mkdir -p $PYDRA_TEST_DATA
datalad install -r -s ///openfmri/ds000114 $PYDRA_TEST_DATA/ds000114
datalad get $PYDRA_TEST_DATA/ds000114/sub-0{1,2}/ses-test/anat/sub-0?_ses-test_T1w.nii.gz
