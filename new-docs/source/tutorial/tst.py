import tempfile
from pathlib import Path
import numpy as np
from fileformats.medimage import Nifti1
import fileformats.medimage_mrtrix3 as mrtrix3
from pydra.engine.environments import Docker
from pydra.design import workflow, python
from pydra.tasks.mrtrix3.v3_0 import MrConvert, MrThreshold

MRTRIX2NUMPY_DTYPES = {
    "Int8": np.dtype("i1"),
    "UInt8": np.dtype("u1"),
    "Int16LE": np.dtype("<i2"),
    "Int16BE": np.dtype(">i2"),
    "UInt16LE": np.dtype("<u2"),
    "UInt16BE": np.dtype(">u2"),
    "Int32LE": np.dtype("<i4"),
    "Int32BE": np.dtype(">i4"),
    "UInt32LE": np.dtype("<u4"),
    "UInt32BE": np.dtype(">u4"),
    "Float32LE": np.dtype("<f4"),
    "Float32BE": np.dtype(">f4"),
    "Float64LE": np.dtype("<f8"),
    "Float64BE": np.dtype(">f8"),
    "CFloat32LE": np.dtype("<c8"),
    "CFloat32BE": np.dtype(">c8"),
    "CFloat64LE": np.dtype("<c16"),
    "CFloat64BE": np.dtype(">c16"),
}


@workflow.define(outputs=["out_image"])
def ToyMedianThreshold(in_image: Nifti1) -> mrtrix3.ImageFormat:
    """A toy example workflow that

    * converts a NIfTI image to MRTrix3 image format with a separate header
    * loads the separate data file and selects the median value
    """

    input_conversion = workflow.add(
        MrConvert(in_file=in_image, out_file="out_file.mih"),
        name="input_conversion",
        environment=Docker("mrtrix3/mrtrix3", tag="latest"),
    )

    @python.define
    def Median(mih: mrtrix3.ImageHeader) -> float:
        """A bespoke function that reads the separate data file in the MRTrix3 image
        header format (i.e. .mih) and calculates the median value."""
        dtype = MRTRIX2NUMPY_DTYPES[mih.metadata["datatype"].strip()]
        data = np.frombuffer(Path.read_bytes(mih.data_file), dtype=dtype)
        return np.median(data)

    median = workflow.add(Median(mih=input_conversion.out_file))
    threshold = workflow.add(
        MrThreshold(in_file=in_image, out_file="binary.mif", abs=median.out),
        environment=Docker("mrtrix3/mrtrix3", tag="latest"),
    )

    output_conversion = workflow.add(
        MrConvert(in_file=threshold.out_file, out_file="out_image.mif"),
        name="output_conversion",
        environment=Docker("mrtrix3/mrtrix3", tag="latest"),
    )

    return output_conversion.out_file


test_dir = tempfile.mkdtemp()

nifti_file = Nifti1.sample(test_dir, seed=0)

wf = ToyMedianThreshold(in_image=nifti_file)

outputs = wf()

print(outputs)
