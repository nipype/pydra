import pydra
import typing as ty
import nest_asyncio
from copy import deepcopy

from nilearn import datasets

import os
import warnings
import load_confounds
import numpy as np
import pandas as pd
import nibabel as nib
from itertools import repeat
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity


def _load_from_strategy(denoiser, fname):
    """Verifies if load_confounds strategy is useable given the regressor files.
    load_confounds will raise it's own exception, but add an additional
    nixtract-specific exception that clarifies the incompatibility.
    """
    error_msg = (
        "load_confound strategy incompatible with provided regressor "
        "files. Check regressor files if they contain the appropriate "
        "columns determined by load_confounds."
    )
    try:
        confounds = denoiser.load(fname)
        return pd.DataFrame(confounds, columns=denoiser.columns_)
    except ValueError as e:
        raise ValueError(error_msg) from e


class FunctionalImage(object):
    def __init__(self, fname):

        self.fname = fname
        img = nib.load(self.fname)
        self.img = img

        self.regressors = None
        self.regressor_file = None

    def set_regressors(self, regressor_fname, regressor_input=None):
        """Set appropriate regressors."""

        # specific strategies for load_confounds
        strategies = [
            "Params2",
            "Params6",
            "Params9",
            "Params24",
            "Params36",
            "AnatCompCor",
            "TempCompCor",
        ]
        flexible_strategies = ["motion", "high_pass", "wm_csf", "compcor", "global"]
        if regressor_fname:
            if regressor_input is None:
                # use all regressors from file
                regressors = pd.read_csv(regressor_fname, sep=r"\t", engine="python")
            elif len(regressor_input) == 1 and (regressor_input[0] in strategies):
                # predefined strategy
                denoiser = eval("load_confounds.{}()".format(regressor_input[0]))
                regressors = _load_from_strategy(denoiser, regressor_fname)
            elif set(regressor_input) <= set(flexible_strategies):
                # flexible strategy
                denoiser = load_confounds.Confounds(strategy=regressor_input)
                regressors = _load_from_strategy(denoiser, regressor_fname)
            elif all(
                [x not in strategies + flexible_strategies for x in regressor_input]
            ):
                # list of regressor names
                try:
                    regressors = pd.read_csv(
                        regressor_fname, sep="\t", usecols=regressor_input
                    )
                except ValueError as e:
                    msg = "Not all regressors are found in regressor file"
                    raise ValueError(msg) from e
            else:
                raise ValueError(
                    "Invalid regressors. Regressors must be a list of "
                    "column names that appear in regressor_files, OR a "
                    "defined load_confounds regressor strategy (flexible "
                    "or non-flexible)."
                )

            self.regressor_file = regressor_fname
            self.regressors = regressors

        elif (self.regressor_file is None) and regressor_input:
            warnings.warn(
                "Regressors are provided without regressor_file. No "
                "confound regression can be done"
            )

    def discard_scans(self, n_scans):
        if n_scans is not None:
            if n_scans > 0:
                # crop scans from image
                arr = self.img.get_data()
                arr = arr[:, :, :, n_scans:]
                self.img = nib.Nifti1Image(arr, self.img.affine)

                if self.regressors is not None:
                    # crop from regressors
                    self.regressors = self.regressors.iloc[n_scans:, :]

    def extract(self, masker, labels=None):
        print("  Extracting from {}".format(os.path.basename(self.fname)))

        timeseries = masker.fit_transform(self.img, confounds=self.regressors.values)

        # determine column names for timeseries
        if isinstance(masker, NiftiMasker):
            labels = ["voxel {}".format(int(i)) for i in np.arange(timeseries.shape[1])]
            self.roi_img = masker.mask_img_
            self.masker_type = "NiftiMasker"

        elif isinstance(masker, NiftiLabelsMasker):
            if labels is None:
                labels = ["roi {}".format(int(i)) for i in masker.labels_]
            self.roi_img = masker.labels_img
            self.masker_type = "NiftiLabelsMasker"

        elif isinstance(masker, NiftiSpheresMasker):
            if labels is None:
                labels = ["roi {}".format(int(i)) for i in range(len(masker.seeds))]
            self.roi_img = masker.spheres_img
            self.masker_type = "NiftiSpheresMasker"

        self.masker = masker
        self.timeseries = pd.DataFrame(timeseries, columns=[str(i) for i in labels])


## MASKING FUNCTIONS
def _get_spheres_from_masker(masker, img):
    """Re-extract spheres from coordinates to make niimg.
    Note that this will take a while, as it uses the exact same function that
    nilearn calls to extract data for NiftiSpheresMasker
    """

    ref_img = nib.load(img)
    ref_img = nib.Nifti1Image(ref_img.get_fdata()[:, :, :, [0]], ref_img.affine)

    X, A = _apply_mask_and_get_affinity(
        masker.seeds, ref_img, masker.radius, masker.allow_overlap
    )
    # label sphere masks
    spheres = A.toarray()
    spheres *= np.arange(1, len(masker.seeds) + 1)[:, np.newaxis]

    # combine masks, taking the maximum if overlap occurs
    arr = np.zeros(spheres.shape[1])
    for i in np.arange(spheres.shape[0]):
        arr = np.maximum(arr, spheres[i, :])
    arr = arr.reshape(ref_img.shape[:-1])
    spheres_img = nib.Nifti1Image(arr, ref_img.affine)

    if masker.mask_img is not None:
        mask_img_ = resample_to_img(masker.mask_img, spheres_img)
        spheres_img = math_img("img1 * img2", img1=spheres_img, img2=mask_img_)

    return spheres_img


def _read_coords(roi_file):
    """Parse and validate coordinates from file"""
    if not roi_file.endswith(".tsv"):
        raise ValueError("Coordinate file must be a tab-separated .tsv file")
    coords = pd.read_table(roi_file)

    # validate columns
    columns = [x for x in coords.columns if x in ["x", "y", "z"]]
    if (len(columns) != 3) or (len(np.unique(columns)) != 3):
        raise ValueError(
            "Provided coordinates do not have 3 columns with names `x`, `y`, and `z`"
        )

    # convert to list of lists for nilearn input
    return coords.values.tolist()


def _masker_from_coords(roi, input_files, output_dir):
    # makes a new mask from the coords and save it
    n_rois = len(roi)
    print("{} region(s) detected from coordinates".format(n_rois))

    # if kwargs.get('radius') is None:
    #    warnings.warn('No radius specified for coordinates; setting to nilearn.input_data.NiftiSphereMasker default of extracting from a single voxel')

    masker = NiftiSpheresMasker(roi)

    # create and save spheres image if coordinates are provided
    masker.spheres_img = _get_spheres_from_masker(masker, input_files[0])
    masker.spheres_img.to_filename(
        os.path.join(output_dir, "nixtract_data", "spheres_image.nii.gz")
    )
    return masker


def _set_masker(roi_file, input_files, output_dir, as_voxels=False):
    """Check and see if multiple ROIs exist in atlas file"""
    # 1) NIfTI image that is an atlas
    # 2) query string formatted as `nilearn:<atlas-name>:<atlas-parameters>
    # 3) a file path to a .tsv file that contains roi_file coordinates in MNI space

    # OPTION 3
    if roi_file.endswith(".tsv"):
        masker = _masker_from_coords(roi, input_files, output_dir)

    # OPTION 1 & 2
    else:
        roi = load_img(roi_file)
        n_rois = len(np.unique(roi.get_data())) - 1
        print("  {} region(s) detected from {}".format(n_rois, roi.get_filename()))

        # if 'radius' in kwargs:
        #    kwargs.pop('radius')
        # if 'allow_overlap' in kwargs:
        #    kwargs.pop('allow_overlap')

        if n_rois == 0:
            raise ValueError("No ROI detected; check ROI file")
        elif (n_rois == 1) & as_voxels:
            #    if 'mask_img' in kwargs:
            #        kwargs.pop('mask_img')
            masker = NiftiMasker(roi)
        else:
            masker = NiftiLabelsMasker(roi)

    return masker


def _gen_img(img_name, regressor_file=None, regressors=None, discard_scans=None):
    img = FunctionalImage(img_name)
    img.set_regressors(regressor_file, regressors)
    img.discard_scans(discard_scans)
    return img


# EXTRACT TIMESERIES


def _save_timeseries(img, output_dir):
    out_fname = os.path.basename(img.fname).split(".")[0] + "_timeseries.tsv"
    img.timeseries.to_csv(
        os.path.join(output_dir, out_fname), sep="\t", index=False, float_format="%.8f"
    )


# GENERATE REPORT


@pydra.mark.task
@pydra.mark.annotate({"return": {"masker": ty.Any}})
def set_masker_pdt(roi_file, input_files, output_dir, as_voxels=False):
    return _set_masker(roi_file, input_files, output_dir, as_voxels=as_voxels)


@pydra.mark.task
@pydra.mark.annotate({"return": {"img": ty.Any}})
def gen_img_pdt(img_name, regressor_file=None, regressors=None, discard_scans=None):
    return _gen_img(
        img_name,
        regressor_file=regressor_file,
        regressors=regressors,
        discard_scans=discard_scans,
    )


@pydra.mark.task
@pydra.mark.annotate({"return": {"img_ts": ty.Any}})
def img_extract_pdt(img, masker, labels=None):
    # dj: removing the original task on img
    # img.extract(masker, labels=labels)
    # return img
    return 3


harvard_oxford = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
data = datasets.fetch_development_fmri(n_subjects=2, reduce_confounds=False)

niifiles = data.func
reg_files = data.confounds
atlas = harvard_oxford.maps
regressors = ["Params6"]


def test_1(tmpdir):
    t1 = set_masker_pdt(roi_file=atlas, input_files=niifiles, output_dir=tmpdir)
    t1()
    masker = t1.result().output.masker


def test_2():
    t2 = gen_img_pdt(img_name=niifiles, regressor_file=reg_files, regressors=regressors)
    t2.split(("img_name", "regressor_file"))

    t2()

    # Keep images for img_extract_pdt ex
    img1 = t2.result()[0].output.img
    img2 = t2.result()[1].output.img


def test_3():
    t3 = gen_img_pdt(
        img_name=niifiles,
        regressor_file=[None, None],  # reg_files,
        regressors=regressors,
    )
    t3.split(("img_name", "regressor_file"))

    t3()
    assert t3.result()


def test_3a():
    t3 = gen_img_pdt(
        img_name=niifiles[0], regressor_file=None, regressors=regressors  # reg_files,
    )
    # t3.split("img_name")

    t3()
    assert t3.result()


def test_4():
    t4 = gen_img_pdt(
        img_name=niifiles, regressor_file=reg_files, regressors=None  # regressors
    )
    t4.split(("img_name", "regressor_file"))

    t4()
    assert t4.result()


def test_5(tmpdir):
    t1 = set_masker_pdt(roi_file=atlas, input_files=niifiles, output_dir=tmpdir)
    t1()
    masker = t1.result().output.masker

    t2 = gen_img_pdt(img_name=niifiles, regressor_file=reg_files, regressors=regressors)
    t2.split(("img_name", "regressor_file"))
    t2()
    # Keep images for img_extract_pdt ex
    img1 = t2.result()[0].output.img
    img2 = t2.result()[1].output.img

    t5 = img_extract_pdt(img=[img1], masker=masker)
    t5.split("img")

    t5()
    assert t5.result()


def test_5a(tmpdir):
    t1 = set_masker_pdt(roi_file=atlas, input_files=niifiles, output_dir=tmpdir)
    t1()
    masker = t1.result().output.masker

    t2 = gen_img_pdt(img_name=niifiles, regressor_file=reg_files, regressors=regressors)
    t2.split(("img_name", "regressor_file"))
    t2()
    # Keep images for img_extract_pdt ex
    img1 = t2.result()[0].output.img

    t5 = img_extract_pdt(img=img1, masker=masker)
    t5()
    assert t5.result()  # .output.img_ts.timeseries
