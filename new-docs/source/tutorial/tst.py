from pathlib import Path
from tempfile import mkdtemp
import shutil
from fileformats.medimage import Nifti1
from pydra.tasks.mrtrix3.v3_0 import MrGrid
from pydra.utils import user_cache_dir


if __name__ == "__main__":
    test_dir = Path(mkdtemp())

    shutil.rmtree(user_cache_dir / "run-cache", ignore_errors=True)

    nifti_dir = test_dir / "nifti"
    nifti_dir.mkdir()

    for i in range(10):
        Nifti1.sample(
            nifti_dir, seed=i
        )  # Create a dummy NIfTI file in the dest. directory

    # Instantiate the task definition, "splitting" over all NIfTI files in the test directory
    # by splitting the "input" input field over all files in the directory
    mrgrid = MrGrid(operation="regrid", voxel=(0.5, 0.5, 0.5)).split(
        in_file=nifti_dir.iterdir()
    )

    # Run the task to resample all NIfTI files
    outputs = mrgrid(worker="cf")

    # Print the locations of the output files
    print("\n".join(str(p) for p in outputs.out_file))
