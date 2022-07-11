"""Functions ported from Nipype 1, after removing parts that were related to py2."""
import attr
import subprocess as sp
from hashlib import sha256
import os
import os.path as op
import re
import shutil
import posixpath
import logging
from pathlib import Path
import typing as ty
from copy import copy

related_filetype_sets = [(".hdr", ".img", ".mat"), (".nii", ".mat"), (".BRIK", ".HEAD")]
"""List of neuroimaging file types that are to be interpreted together."""

logger = logging.getLogger("pydra")


def split_filename(fname):
    """
    Split a filename into parts: path, base filename and extension.

    Parameters
    ----------
    fname : :obj:`str`
        file or path name

    Returns
    -------
    pth : :obj:`str`
        base path from fname
    fname : :obj:`str`
        filename from fname, without extension
    ext : :obj:`str`
        file extension from fname

    Examples
    --------
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'

    >>> fname
    'subject'

    >>> ext
    '.nii.gz'

    """
    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]

    pth = op.dirname(fname)
    fname = op.basename(fname)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = op.splitext(fname)

    return pth, fname, ext


def hash_file(
    afile, chunk_len=8192, crypto=sha256, raise_notfound=True, precalculated=None
):
    """Compute hash of a file using 'crypto' module."""
    from .specs import LazyField

    if afile is None or isinstance(afile, LazyField) or isinstance(afile, list):
        return None
    if not Path(afile).is_file():
        if raise_notfound:
            raise RuntimeError('File "%s" not found.' % afile)
        return None

    # if the path exists already in precalculated
    # the time of the last modification will be compared
    # and the precalculated hash value will be used if the file has not change
    if precalculated and str(Path(afile)) in precalculated:
        pre_mtime, pre_cont_hash = precalculated[str(Path(afile))]
        if Path(afile).stat().st_mtime == pre_mtime:
            return pre_cont_hash

    crypto_obj = crypto()
    with open(afile, "rb") as fp:
        while True:
            data = fp.read(chunk_len)
            if not data:
                break
            crypto_obj.update(data)

    cont_hash = crypto_obj.hexdigest()
    if precalculated is not None:
        precalculated[str(Path(afile))] = (Path(afile).stat().st_mtime, cont_hash)
    return cont_hash


def hash_dir(
    dirpath,
    crypto=sha256,
    ignore_hidden_files=False,
    ignore_hidden_dirs=False,
    raise_notfound=True,
    precalculated=None,
):
    """Compute hash of directory contents.

    This function computes the hash of every file in directory `dirpath` and then
    computes the hash of that list of hashes to return a single hash value. The
    directory is traversed recursively.

    Parameters
    ----------
    dirpath : :obj:`str`
        Path to directory.
    crypto : :obj: `function`
        cryptographic hash functions
    ignore_hidden_files : :obj:`bool`
        If `True`, ignore filenames that begin with `.`.
    ignore_hidden_dirs : :obj:`bool`
        If `True`, ignore files in directories that begin with `.`.
    raise_notfound : :obj:`bool`
        If `True` and `dirpath` does not exist, raise `FileNotFound` exception. If
        `False` and `dirpath` does not exist, return `None`.

    Returns
    -------
    hash : :obj:`str`
        Hash of the directory contents.
    """
    from .specs import LazyField

    if dirpath is None or isinstance(dirpath, LazyField) or isinstance(dirpath, list):
        return None
    if not Path(dirpath).is_dir():
        if raise_notfound:
            raise FileNotFoundError(f"Directory {dirpath} not found.")
        return None

    file_hashes = []
    for dpath, dirnames, filenames in os.walk(dirpath):
        # Sort in-place to guarantee order.
        dirnames.sort()
        filenames.sort()
        dpath = Path(dpath)
        if ignore_hidden_dirs and dpath.name.startswith(".") and str(dpath) != dirpath:
            continue
        for filename in filenames:
            if ignore_hidden_files and filename.startswith("."):
                continue
            if not is_existing_file(dpath / filename):
                file_hashes.append(str(dpath / filename))
            else:
                this_hash = hash_file(dpath / filename, precalculated=precalculated)
                file_hashes.append(this_hash)

    crypto_obj = crypto()
    for h in file_hashes:
        crypto_obj.update(h.encode())

    return crypto_obj.hexdigest()


def _parse_mount_table(exit_code, output):
    """
    Parse the output of ``mount`` to produce (path, fs_type) pairs.

    Separated from _generate_cifs_table to enable testing logic with real
    outputs

    """
    # Not POSIX
    if exit_code != 0:
        return []

    # Linux mount example:  sysfs on /sys type sysfs (rw,nosuid,nodev,noexec)
    #                          <PATH>^^^^      ^^^^^<FSTYPE>
    # OSX mount example:    /dev/disk2 on / (hfs, local, journaled)
    #                               <PATH>^  ^^^<FSTYPE>
    pattern = re.compile(r".*? on (/.*?) (?:type |\()([^\s,\)]+)")

    # Keep line and match for error reporting (match == None on failure)
    # Ignore empty lines
    matches = [(ll, pattern.match(ll)) for ll in output.strip().splitlines() if ll]

    # (path, fstype) tuples, sorted by path length (longest first)
    mount_info = sorted(
        (match.groups() for _, match in matches if match is not None),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    cifs_paths = [path for path, fstype in mount_info if fstype.lower() == "cifs"]

    # Report failures as warnings
    for line, match in matches:
        if match is None:
            logger.debug("Cannot parse mount line: '%s'", line)

    return [
        mount
        for mount in mount_info
        if any(mount[0].startswith(path) for path in cifs_paths)
    ]


def _generate_cifs_table():
    """
    Construct a reverse-length-ordered list of mount points that fall under a CIFS mount.

    This precomputation allows efficient checking for whether a given path
    would be on a CIFS filesystem.
    On systems without a ``mount`` command, or with no CIFS mounts, returns an
    empty list.

    """
    exit_code, output = sp.getstatusoutput("mount")
    return _parse_mount_table(exit_code, output)


_cifs_table = _generate_cifs_table()


def on_cifs(fname):
    """
    Check whether a file path is on a CIFS filesystem mounted in a POSIX host.

    POSIX hosts are assumed to have the ``mount`` command.

    On Windows, Docker mounts host directories into containers through CIFS
    shares, which has support for Minshall+French symlinks, or text files that
    the CIFS driver exposes to the OS as symlinks.
    We have found that under concurrent access to the filesystem, this feature
    can result in failures to create or read recently-created symlinks,
    leading to inconsistent behavior and ``FileNotFoundError`` errors.

    This check is written to support disabling symlinks on CIFS shares.

    """
    # Only the first match (most recent parent) counts
    for fspath, fstype in _cifs_table:
        if fname.startswith(fspath):
            return fstype == "cifs"
    return False


def copyfile(
    originalfile,
    newfile,
    copy=False,
    create_new=False,
    use_hardlink=True,
    copy_related_files=True,
):
    """
    Copy or link files.

    If ``use_hardlink`` is True, and the file can be hard-linked, then a
    link is created, instead of copying the file.

    If a hard link is not created and ``copy`` is False, then a symbolic
    link is created.

    .. admonition:: Copy options for existing files

        * symlink

            * to regular file originalfile            (keep if symlinking)
            * to same dest as symlink originalfile    (keep if symlinking)
            * to other file                           (unlink)

        * regular file

            * hard link to originalfile               (keep)
            * copy of file (same hash)                (keep)
            * different file (diff hash)              (unlink)

    .. admonition:: Copy options for new files

        * ``use_hardlink`` & ``can_hardlink`` => hardlink
        * ``~hardlink`` & ``~copy`` & ``can_symlink`` => symlink
        * ``~hardlink`` & ``~symlink`` => copy

    Parameters
    ----------
    originalfile : :obj:`str`
        full path to original file
    newfile : :obj:`str`
        full path to new file
    copy : Bool
        specifies whether to copy or symlink files
        (default=False) but only for POSIX systems
    use_hardlink : Bool
        specifies whether to hard-link files, when able
        (Default=False), taking precedence over copy
    copy_related_files : Bool
        specifies whether to also operate on related files, as defined in
        ``related_filetype_sets``

    Returns
    -------
    None

    """
    newhash = None
    orighash = None
    logger.debug(newfile)

    if create_new:
        while op.exists(newfile):
            base, fname, ext = split_filename(newfile)
            s = re.search("_c[0-9]{4,4}$", fname)
            i = 0
            if s:
                i = int(s.group()[2:]) + 1
                fname = fname[:-6] + "_c%04d" % i
            else:
                fname += "_c%04d" % i
            newfile = base + os.sep + fname + ext

    # Don't try creating symlinks on CIFS
    if copy is False and on_cifs(newfile):
        copy = True

    keep = False
    if op.lexists(newfile):
        if op.islink(newfile):
            if all(
                (
                    os.readlink(newfile) == op.realpath(originalfile),
                    not use_hardlink,
                    not copy,
                )
            ):
                keep = True
        elif posixpath.samefile(newfile, originalfile):
            keep = True
        else:
            newhash = hash_file(newfile)
            logger.debug("File: %s already exists,%s, copy:%d", newfile, newhash, copy)
            orighash = hash_file(originalfile)
            keep = newhash == orighash
        if keep:
            logger.debug(
                "File: %s already exists, not overwriting, copy:%d", newfile, copy
            )
        else:
            os.unlink(newfile)

    if not keep and use_hardlink:
        try:
            logger.debug("Linking File: %s->%s", newfile, originalfile)
            # Use realpath to avoid hardlinking symlinks
            os.link(op.realpath(originalfile), newfile)
        except OSError:
            use_hardlink = False  # Disable hardlink for associated files
        else:
            keep = True

    if not keep and not copy and os.name == "posix":
        try:
            logger.debug("Symlinking File: %s->%s", newfile, originalfile)
            os.symlink(originalfile, newfile)
        except OSError:
            copy = True  # Disable symlink for associated files
        else:
            keep = True

    if not keep:
        try:
            logger.debug("Copying File: %s->%s", newfile, originalfile)
            shutil.copyfile(originalfile, newfile)
        except shutil.Error as e:
            logger.warning(e.message)

    # Associated files
    if copy_related_files:
        related_file_pairs = (
            get_related_files(f, include_this_file=False)
            for f in (originalfile, newfile)
        )
        for alt_ofile, alt_nfile in zip(*related_file_pairs):
            if op.exists(alt_ofile):
                copyfile(
                    alt_ofile,
                    alt_nfile,
                    copy,
                    use_hardlink=use_hardlink,
                    copy_related_files=False,
                )

    return newfile


def get_related_files(filename, include_this_file=True):
    """
    Return a list of related files.

    As defined in :attr:`related_filetype_sets`, for a filename
    (e.g., Nifti-Pair, Analyze (SPM), and AFNI files).

    Parameters
    ----------
    filename : :obj:`str`
        File name to find related filetypes of.
    include_this_file : bool
        If true, output includes the input filename.

    """
    related_files = []
    path, name, this_type = split_filename(filename)
    for type_set in related_filetype_sets:
        if this_type in type_set:
            for related_type in type_set:
                if include_this_file or related_type != this_type:
                    related_files.append(Path(path) / (name + related_type))
    if not len(related_files):
        related_files = [filename]
    return related_files


def copyfiles(filelist, dest, copy=False, create_new=False):
    """
    Copy or symlink files in ``filelist`` to ``dest`` directory.

    Parameters
    ----------
    filelist : list
        List of files to copy.
    dest : path/files
        full path to destination. If it is a list of length greater
        than 1, then it assumes that these are the names of the new
        files.
    copy : Bool
        specifies whether to copy or symlink files
        (default=False) but only for posix systems

    Returns
    -------
    None

    """
    # checking if dest is a single dir or filepath/filepaths
    if not isinstance(dest, list) and Path(dest).is_dir():
        dest_dir = True
        out_path = str(Path(dest).resolve())
    else:
        dest_dir = False
        out_path = ensure_list(dest)
    newfiles = []
    for i, f in enumerate(ensure_list(filelist)):
        # Todo: this part is not tested
        if isinstance(f, list):
            newfiles.insert(i, copyfiles(f, dest, copy=copy, create_new=create_new))
        else:
            if dest_dir:
                destfile = fname_presuffix(f, newpath=out_path)
            else:
                destfile = out_path[i]
            destfile = copyfile(f, destfile, copy, create_new=create_new)
            newfiles.insert(i, destfile)
    return newfiles


def fname_presuffix(fname, prefix="", suffix="", newpath=None, use_ext=True):
    """
    Manipulate path and name of input filename.

    Parameters
    ----------
    fname : :obj:`str`
        A filename (may or may not include path)
    prefix : :obj:`str`
        Characters to prepend to the filename
    suffix : :obj:`str`
        Characters to append to the filename
    newpath : :obj:`str`
        Path to replace the path of the input fname
    use_ext : :obj:`bool`
        If True (default), appends the extension of the original file
        to the output name.
    Return
    ------
    path : :obj:`str`
        Absolute path of the modified filename
    Examples
    --------
    >>> import pytest, sys
    >>> if sys.platform.startswith('win'): pytest.skip()
    >>> from pydra.engine.helpers_file import fname_presuffix
    >>> fname = 'foo.nii.gz'
    >>> fname_presuffix(fname,'pre','post','/tmp')
    '/tmp/prefoopost.nii.gz'
    """
    pth, fname, ext = split_filename(fname)
    if not use_ext:
        ext = ""

    # No need for isdefined: bool(Undefined) evaluates to False
    if newpath:
        pth = op.abspath(newpath)
    return str(Path(pth) / (prefix + fname + suffix + ext))


# dj: copied from misc
def is_container(item):
    """
    Check if item is a container (list, tuple, dict, set).

    Parameters
    ----------
    item : :obj:`object`
        Input object to check.

    Returns
    -------
    output : :obj:`bool`
        ``True`` if container ``False`` otherwise.

    """
    if isinstance(item, str):
        return False
    elif hasattr(item, "__iter__"):
        return True

    return False


def ensure_list(filename):
    """Return a list given either a string or a list."""
    if isinstance(filename, (str, bytes)):
        return [filename]
    elif isinstance(filename, list):
        return filename
    elif is_container(filename):
        return [x for x in filename]

    return None


# not sure if this might be useful for Function Task
def copyfile_input(inputs, output_dir):
    """Implement the base class method."""
    from .specs import attr_fields, File, MultiInputFile

    map_copyfiles = {}
    for fld in attr_fields(inputs):
        copy = fld.metadata.get("copyfile")
        if copy is not None and fld.type not in [File, MultiInputFile]:
            raise Exception(
                f"if copyfile set, field has to be a File " f"but {fld.type} provided"
            )
        file = getattr(inputs, fld.name)
        if copy in [True, False] and file != attr.NOTHING:
            if isinstance(file, list):
                map_copyfiles[fld.name] = []
                for el in file:
                    newfile = output_dir.joinpath(Path(el).name)
                    copyfile(el, newfile, copy=copy)
                    map_copyfiles[fld.name].append(str(newfile))
            else:
                newfile = output_dir.joinpath(Path(file).name)
                copyfile(file, newfile, copy=copy)
                map_copyfiles[fld.name] = str(newfile)
    return map_copyfiles or None


# not sure if this might be useful for Function Task
def template_update(inputs, output_dir, state_ind=None, map_copyfiles=None):
    """
    Update all templates that are present in the input spec.

    Should be run when all inputs used in the templates are already set.

    """

    inputs_dict_st = attr.asdict(inputs, recurse=False)
    if map_copyfiles is not None:
        inputs_dict_st.update(map_copyfiles)

    if state_ind is not None:
        for k, v in state_ind.items():
            k = k.split(".")[1]
            inputs_dict_st[k] = inputs_dict_st[k][v]

    from .specs import attr_fields

    fields_templ = [
        fld for fld in attr_fields(inputs) if fld.metadata.get("output_file_template")
    ]
    dict_mod = {}
    for fld in fields_templ:
        if fld.type not in [str, ty.Union[str, bool]]:
            raise Exception(
                f"fields with output_file_template"
                " has to be a string or Union[str, bool]"
            )
        dict_mod[fld.name] = template_update_single(
            field=fld,
            inputs=inputs,
            inputs_dict_st=inputs_dict_st,
            output_dir=output_dir,
        )
    # adding elements from map_copyfiles to fields with templates
    if map_copyfiles:
        dict_mod.update(map_copyfiles)
    return dict_mod


def template_update_single(
    field, inputs, inputs_dict_st=None, output_dir=None, spec_type="input"
):
    """Update a single template from the input_spec or output_spec
    based on the value from inputs_dict
    (checking the types of the fields, that have "output_file_template)"
    """
    from .specs import File, MultiOutputFile, Directory

    # if input_dict_st with state specific value is not available,
    # the dictionary will be created from inputs object
    if inputs_dict_st is None:
        inputs_dict_st = attr.asdict(inputs, recurse=False)

    if spec_type == "input":
        if field.type not in [str, ty.Union[str, bool]]:
            raise Exception(
                f"fields with output_file_template"
                "has to be a string or Union[str, bool]"
            )
        inp_val_set = inputs_dict_st[field.name]
        if inp_val_set is not attr.NOTHING and not isinstance(inp_val_set, (str, bool)):
            raise Exception(
                f"{field.name} has to be str or bool, but {inp_val_set} set"
            )
        if isinstance(inp_val_set, bool) and field.type is str:
            raise Exception(
                f"type of {field.name} is str, consider using Union[str, bool]"
            )
    elif spec_type == "output":
        if field.type not in [File, MultiOutputFile, Directory]:
            raise Exception(
                f"output {field.name} should be a File, but {field.type} set as the type"
            )
    else:
        raise Exception(f"spec_type can be input or output, but {spec_type} provided")
    # for inputs that the value is set (so the template is ignored)
    if spec_type == "input" and isinstance(inputs_dict_st[field.name], str):
        return inputs_dict_st[field.name]
    elif spec_type == "input" and inputs_dict_st[field.name] is False:
        # if input fld is set to False, the fld shouldn't be used (setting NOTHING)
        return attr.NOTHING
    else:  # inputs_dict[field.name] is True or spec_type is output
        value = _template_formatting(field, inputs, inputs_dict_st)
        # changing path so it is in the output_dir
        if output_dir and value is not attr.NOTHING:
            # should be converted to str, it is also used for input fields that should be str
            if type(value) is list:
                return [str(output_dir / Path(val).name) for val in value]
            else:
                return str(output_dir / Path(value).name)
        else:
            return attr.NOTHING


def _template_formatting(field, inputs, inputs_dict_st):
    """Formatting the field template based on the values from inputs.
    Taking into account that the field with a template can be a MultiOutputFile
    and the field values needed in the template can be a list -
    returning a list of formatted templates in that case.
    Allowing for multiple input values used in the template as longs as
    there is no more than one file (i.e. File, PathLike or string with extensions)
    """
    from .specs import MultiOutputFile

    # if a template is a function it has to be run first with the inputs as the only arg
    template = field.metadata["output_file_template"]
    if callable(template):
        template = template(inputs)

    # as default, we assume that keep_extension is True
    keep_extension = field.metadata.get("keep_extension", True)

    inp_fields = re.findall(r"{\w+}", template)
    inp_fields_fl = re.findall(r"{\w+:[0-9.]+f}", template)
    inp_fields += [re.sub(":[0-9.]+f", "", el) for el in inp_fields_fl]
    if len(inp_fields) == 0:
        return template

    val_dict = {}
    file_template = None
    from .specs import attr_fields_dict, File

    for fld in inp_fields:
        fld_name = fld[1:-1]  # extracting the name form {field_name}
        if fld_name not in inputs_dict_st:
            raise AttributeError(f"{fld_name} is not provided in the input")
        fld_value = inputs_dict_st[fld_name]
        if fld_value is attr.NOTHING:
            # if value is NOTHING, nothing should be added to the command
            return attr.NOTHING
        else:
            # checking for fields that can be treated as a file:
            # have type File, or value that is path like (including str with extensions)
            if (
                attr_fields_dict(inputs)[fld_name].type is File
                or isinstance(fld_value, os.PathLike)
                or (isinstance(fld_value, str) and "." in fld_value)
            ):
                if file_template:
                    raise Exception(
                        f"can't have multiple paths in {field.name} template,"
                        f" but {template} provided"
                    )
                else:
                    file_template = (fld_name, fld_value)
            else:
                val_dict[fld_name] = fld_value

    # if field is MultiOutputFile and some elements from val_dict are lists,
    # each element of the list should be used separately in the template
    # and return a list with formatted values
    if field.type is MultiOutputFile and any(
        [isinstance(el, list) for el in val_dict.values()]
    ):
        # all fields that are lists
        keys_list = [k for k, el in val_dict.items() if isinstance(el, list)]
        if any(
            [len(val_dict[key]) != len(val_dict[keys_list[0]]) for key in keys_list[1:]]
        ):
            raise Exception(
                f"all fields used in {field.name} template have to have the same length"
                f" or be a single value"
            )
        formatted_value = []
        for ii in range(len(val_dict[keys_list[0]])):
            val_dict_el = copy(val_dict)
            # updating values to a single element from the list
            for key in keys_list:
                val_dict_el[key] = val_dict[key][ii]

            formatted_value.append(
                _element_formatting(
                    template, val_dict_el, file_template, keep_extension=keep_extension
                )
            )
    else:
        formatted_value = _element_formatting(
            template, val_dict, file_template, keep_extension=keep_extension
        )
    return formatted_value


def _element_formatting(template, values_template_dict, file_template, keep_extension):
    """Formatting a single template for a single element (if a list).
    Taking into account that a file used in the template (file_template)
    and the template itself could have file extensions
    (assuming that if template has extension, the field value extension is removed,
    if field has extension, and no template extension, than it is moved to the end).
    For values_template_dict the simple formatting can be used (no file values inside)
    """
    if file_template:
        fld_name_file, fld_value_file = file_template
        # splitting the filename for name and extension,
        # the final value used for formatting depends on the template and keep_extension flag
        name, *ext = Path(fld_value_file).name.split(".", maxsplit=1)
        filename = str(Path(fld_value_file).parent / name)
        # updating values_template_dic with the name of file
        values_template_dict[fld_name_file] = filename
        # if keep_extension is False, the extensions are removed
        if keep_extension is False:
            ext = []
    else:
        ext = []

    # if file_template is at the end of the template, the simplest formatting should work
    if file_template and template.endswith(f"{{{fld_name_file}}}"):
        # recreating fld_value with the updated extension
        values_template_dict[fld_name_file] = ".".join([filename] + ext)
        formatted_value = template.format(**values_template_dict)
    # file_template provided, but the template doesn't have its own extension
    elif file_template and "." not in template:
        # if the fld_value_file has extension, it will be moved to the end
        formatted_value = ".".join([template.format(**values_template_dict)] + ext)
    # template has its own extension or no file_template provided
    # the simplest formatting, if file_template is provided it's used without the extension
    else:
        formatted_value = template.format(**values_template_dict)
    return formatted_value


def is_local_file(f):
    from .specs import File, Directory, MultiInputFile

    if "container_path" not in f.metadata and (
        f.type in [File, Directory, MultiInputFile]
        or "pydra.engine.specs.File" in str(f.type)
        or "pydra.engine.specs.Directory" in str(f.type)
    ):
        return True
    else:
        return False


def is_existing_file(value):
    """checking if an object is an existing file"""
    if isinstance(value, str) and value == "":
        return False
    try:
        return Path(value).exists()
    except TypeError:
        return False
