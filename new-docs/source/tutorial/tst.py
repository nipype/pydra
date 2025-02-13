from pathlib import Path
from tempfile import mkdtemp
from pprint import pprint
import json
from pydra.utils.hash import hash_function
from pydra.tasks.mrtrix3.v3_0 import MrGrid
from fileformats.medimage import Nifti1

JSON_CONTENTS = {"a": True, "b": "two", "c": 3, "d": [7, 0.55, 6]}

test_dir = Path(mkdtemp())
cache_root = Path(mkdtemp())
json_file = test_dir / "test.json"
with open(json_file, "w") as f:
    json.dump(JSON_CONTENTS, f)

nifti_dir = test_dir / "nifti"
nifti_dir.mkdir()

for i in range(10):
    Nifti1.sample(nifti_dir, seed=i)  # Create a dummy NIfTI file in the dest. directory

niftis = list(nifti_dir.iterdir())
pprint([hash_function(nifti) for nifti in niftis])

mrgrid_varying_vox_sizes = MrGrid(operation="regrid").split(
    ("in_file", "voxel"),
    in_file=niftis,
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

outputs = mrgrid_varying_vox_sizes(cache_dir=cache_root)

pprint(outputs.out_file)
