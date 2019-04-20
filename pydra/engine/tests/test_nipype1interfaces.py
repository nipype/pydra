import os
from pathlib import Path
import pytest, pdb

from ..node import Workflow
from ..submitter import Submitter

from nipype.interfaces import fsl

Plugins = ["serial", "cf"]

TEST_DATA_DIR = Path(os.getenv("PYDRA_TEST_DATA", "/nonexistent/path"))
DS114_DIR = TEST_DATA_DIR / "ds000114"

@pytest.fixture(scope="module")
def change_dir(request):
    orig_dir = os.getcwd()
    test_dir = os.path.join(orig_dir, "test_nipype1_outputs")
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    def move2orig():
        os.chdir(orig_dir)

    request.addfinalizer(move2orig)


# testing CurrentInterface that is a temporary wrapper for current interfaces
T1_file = "/Users/dorota/nipype_workshop/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"
T1_file_list = [
    "/Users/dorota/nipype_workshop/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz",
    "/Users/dorota/nipype_workshop/data/ds000114/sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz",
]


@pytest.mark.skip("WIP")
@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
def test_current_node_1(change_dir, plugin):
    """Node with a current interface and inputs, no splitter, running interface"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")
    nn = Node(
        name="NA",
        inputs={
            "in_file": str(
                DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"
            )
        },
        interface=interf_bet,
        workingdir="test_cnd1_{}".format(plugin),
        output_names=["out_file"],
    )

    with Submitter(plugin=plugin) as sub:
        sub.run(nn)
    # TODO (res): nodes only returns relative path
    assert "out_file" in nn.output.keys()


@pytest.mark.skip("WIP")
@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
def test_current_node_2(change_dir, plugin):
    """Node with a current interface and splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    in_file_l = [
        str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"),
        str(DS114_DIR / "sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz"),
    ]
    nn = Node(
        name="NA",
        inputs={"in_file": in_file_l},
        splitter="in_file",
        interface=interf_bet,
        write_state=False,
        workingdir="test_cnd2_{}".format(plugin),
        output_names=["out_file"],
    )

    with Submitter(plugin=plugin) as sub:
        sub.run(nn)

    assert "out_file" in nn.output.keys()
    assert "NA.in_file:0" in nn.output["out_file"].keys()
    assert "NA.in_file:1" in nn.output["out_file"].keys()


@pytest.mark.skip("WIP")
@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
def test_current_wf_1(change_dir, plugin):
    """Wf with a current interface, no splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    nn = Node(
        name="fsl",
        inputs={
            "in_file": str(
                DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"
            )
        },
        interface=interf_bet,
        workingdir="nn",
        output_names=["out_file"],
        write_state=False,
    )

    wf = Workflow(
        workingdir="test_cwf_1_{}".format(plugin),
        name="cw1",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False,
    )
    wf.add_nodes([nn])

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    assert "fsl_out" in wf.output.keys()


@pytest.mark.skip("WIP")
@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
def test_current_wf_1a(change_dir, plugin):
    """Wf with a current interface, no splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    nn = Node(
        name="fsl",
        inputs={
            "in_file": str(
                DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"
            )
        },
        interface=interf_bet,
        workingdir="nn",
        output_names=["out_file"],
        write_state=False,
    )

    wf = Workflow(
        workingdir="test_cwf_1a_{}".format(plugin),
        name="cw1",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False,
    )
    wf.add(runnable=nn)

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    assert "fsl_out" in wf.output.keys()


@pytest.mark.skip("WIP")
@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
def test_current_wf_1b(change_dir, plugin):
    """Wf with a current interface, no splitter; using wf.add(nipype CurrentInterface)"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    wf = Workflow(
        workingdir="test_cwf_1b_{}".format(plugin),
        name="cw1",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False,
    )
    wf.add(
        runnable=interf_bet,
        name="fsl",
        workingdir="nn",
        output_names=["out_file"],
        write_state=False,
        inputs={
            "in_file": str(
                DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"
            )
        },
    )

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    assert "fsl_out" in wf.output.keys()


@pytest.mark.skip("WIP")
@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
def test_current_wf_1c(change_dir, plugin):
    """Wf with a current interface, no splitter; using wf.add(nipype interface) """

    wf = Workflow(
        workingdir="test_cwf_1c_{}".format(plugin),
        name="cw1",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False,
    )
    wf.add(
        runnable=fsl.BET(),
        name="fsl",
        workingdir="nn",
        output_names=["out_file"],
        write_state=False,
        inputs={
            "in_file": str(
                DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"
            )
        },
    )

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    assert "fsl_out" in wf.output.keys()


@pytest.mark.skip("WIP")
@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
def test_current_wf_2(change_dir, plugin):
    """Wf with a current interface and splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    in_file_l = [
        str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"),
        str(DS114_DIR / "sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz"),
    ]

    nn = Node(
        name="fsl",
        interface=interf_bet,
        write_state=False,
        workingdir="nn",
        output_names=["out_file"],
    )

    wf = Workflow(
        workingdir="test_cwf_2_{}".format(plugin),
        name="cw2",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        inputs={"in_file": in_file_l},
        splitter="in_file",
        write_state=False,
    )
    wf.add_nodes([nn])
    wf.connect_wf_input("in_file", "fsl", "in_file")

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    assert "fsl_out" in wf.output.keys()
    assert "cw2.in_file:0" in wf.output["fsl_out"].keys()
    assert "cw2.in_file:1" in wf.output["fsl_out"].keys()


@pytest.mark.skip("WIP")
@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
def test_current_wf_2a(change_dir, plugin):
    """Wf with a current interface and splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    in_file_l = [
        str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"),
        str(DS114_DIR / "sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz"),
    ]

    nn = Node(
        name="fsl",
        interface=interf_bet,
        write_state=False,
        workingdir="nn",
        output_names=["out_file"],
        inputs={"in_file": in_file_l},
        splitter="in_file",
    )

    wf = Workflow(
        workingdir="test_cwf_2a_{}".format(plugin),
        name="cw2a",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False,
    )
    wf.add_nodes([nn])
    # wf.connect_wf_input("in_file", "fsl", "in_file")

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    assert "fsl_out" in wf.output.keys()
    assert "fsl.in_file:0" in wf.output["fsl_out"].keys()
    assert "fsl.in_file:1" in wf.output["fsl_out"].keys()
