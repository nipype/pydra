import tempfile
import numpy as np
from fileformats.medimage import Nifti1
import fileformats.medimage_mrtrix3 as mrtrix3
from pydra.engine.environments import Docker
from pydra.design import workflow, python
from pydra.tasks.mrtrix3.v3_0 import MrConvert, MrThreshold


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
    def SelectDataFile(in_file: mrtrix3.ImageHeader) -> mrtrix3.ImageDataFile:
        return in_file.data_file

    select_data = workflow.add(SelectDataFile(in_file=input_conversion.out_file))

    @python.define
    def Median(data_file: mrtrix3.ImageDataFile) -> float:
        data = np.load(data_file)
        return np.median(data)

    median = workflow.add(Median(data_file=select_data.out))
    threshold = workflow.add(
        MrThreshold(in_file=in_image, abs=median.out),
        environment=Docker("mrtrix3/mrtrix3", tag=""),
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
