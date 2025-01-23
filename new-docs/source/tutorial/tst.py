from pathlib import Path
from tempfile import mkdtemp
from fileformats.medimage import Nifti1
from pydra.tasks.mrtrix3.v3_0 import MrGrid


if __name__ == "__main__":

    test_dir = Path(mkdtemp())

    nifti_dir = test_dir / "nifti"
    nifti_dir.mkdir()

    for i in range(10):
        Nifti1.sample(
            nifti_dir, seed=i
        )  # Create a dummy NIfTI file in the dest. directory

    mrgrid_varying_vox_sizes = MrGrid(operation="regrid").split(
        ("in_file", "voxel"),
        in_file=nifti_dir.iterdir(),
        # Define a list of voxel sizes to resample the NIfTI files to,
        # the list must be the same length as the list of NIfTI files
        voxel=[
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.75, 0.75, 0.75),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (1.25, 1.25, 1.25),
            (1.25, 1.25, 1.25),
        ],
    )

print(mrgrid_varying_vox_sizes().out_file)
