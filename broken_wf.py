from pathlib import Path
import logging
from fileformats.generic import Directory
from fileformats.medimage import NiftiGz
from pydra import Submitter, Workflow
from pydra.mark import annotate, task


# Producer of T1w images
@task
@annotate({"return": {"t1w_images": list[Path]}})
def read_t1w_images(bids_dir: Directory) -> list[Path]:
    return list(bids_dir.rglob("*T1w.nii.gz"))


# Mapped to each T1w image
@task
@annotate({"return": {"smoothed_image": Path}})
def smooth_image(input_image: NiftiGz, smoothed_image: Path) -> Path:
    from nilearn.image import load_img, smooth_img

    smoothed_image = smoothed_image or Path.cwd() / (
        input_image.name.split(".", maxsplit=1)[0] + "_smoothed.nii.gz"
    )

    smooth_img(load_img(input_image), fwhm=3).to_filename(smoothed_image)

    return smoothed_image


# Workflow composing both tasks
wf = Workflow(
    name="a_workflow",
    input_spec=["bids_dir"],
    bids_dir="/Users/tclose/Data/openneuro/ds000114",
)
wf.add(read_t1w_images(name="read", bids_dir=wf.lzin.bids_dir))
wf.add(
    smooth_image(name="smooth").split(
        "input_image", input_image=wf.read.lzout.t1w_images
    )
)
wf.set_output({"smoothed_images": wf.smooth.lzout.smoothed_image})

logger = logging.getLogger("pydra")
logger.setLevel(logging.DEBUG)

logging.basicConfig(filename="/Users/tclose/Desktop/test.log", level=logging.DEBUG)

# Run workflow
with Submitter() as sub:
    res = sub(wf)
