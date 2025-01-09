from pathlib import Path
from tempfile import mkdtemp
from fileformats.medimage import Nifti
from pydra.tasks.mrtrix3.v3_0 import MrGrid


if __name__ == "__main__":
    test_dir = Path(mkdtemp())

    nifti_dir = test_dir / "nifti"
    nifti_dir.mkdir()

    for i in range(10):
        Nifti.sample(
            nifti_dir, seed=i
        )  # Create a dummy NIfTI file in the dest. directory

    # Instantiate the task definition, "splitting" over all NIfTI files in the test directory
    # by splitting the "input" input field over all files in the directory
    mrgrid = MrGrid(voxel=(0.5, 0.5, 0.5)).split(input=nifti_dir.iterdir())

    # Run the task to resample all NIfTI files
    outputs = mrgrid()

    # Print the locations of the output files
    print("\n".join(str(p) for p in outputs.output))
