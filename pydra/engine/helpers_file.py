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


def hash_file(afile, chunk_len=8192, crypto=sha256, raise_notfound=True):
    """Compute hash of a file using 'crypto' module."""
    from .specs import LazyField

    if afile is None or isinstance(afile, LazyField) or isinstance(afile, list):
        return None
    if not Path(afile).is_file():
        if raise_notfound:
            raise RuntimeError('File "%s" not found.' % afile)
        return None

    crypto_obj = crypto()
    with open(afile, "rb") as fp:
        while True:
            data = fp.read(chunk_len)
            if not data:
                break
            crypto_obj.update(data)
    return crypto_obj.hexdigest()


def hash_dir(
    dirpath,
    crypto=sha256,
    ignore_hidden_files=False,
    ignore_hidden_dirs=False,
    raise_notfound=True,
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
                this_hash = hash_file(dpath / filename)
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
    matches = [(l, pattern.match(l)) for l in output.strip().splitlines() if l]

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
    outfiles = ensure_list(dest)
    newfiles = []
    for i, f in enumerate(ensure_list(filelist)):
        if isinstance(f, list):
            newfiles.insert(i, copyfiles(f, dest, copy=copy, create_new=create_new))
        else:
            if len(outfiles) > 1:
                destfile = outfiles[i]
            else:
                destfile = fname_presuffix(f, newpath=outfiles[0])
            destfile = copyfile(f, destfile, copy, create_new=create_new)
            newfiles.insert(i, destfile)
    return newfiles


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
    from .specs import attr_fields, File

    map_copyfiles = {}
    for fld in attr_fields(inputs):
        copy = fld.metadata.get("copyfile")
        if copy is not None and fld.type is not File:
            raise Exception(
                f"if copyfile set, field has to be a File " f"but {fld.type} provided"
            )
        if copy in [True, False]:
            file = getattr(inputs, fld.name)
            newfile = output_dir.joinpath(Path(getattr(inputs, fld.name)).name)
            copyfile(file, newfile, copy=copy)
            map_copyfiles[fld.name] = str(newfile)
    return map_copyfiles or None


# not sure if this might be useful for Function Task
def template_update(inputs, output_dir, map_copyfiles=None):
    """
    Update all templates that are present in the input spec.

    Should be run when all inputs used in the templates are already set.

    """
    dict_ = attr.asdict(inputs)
    if map_copyfiles is not None:
        dict_.update(map_copyfiles)

    from .specs import attr_fields

    fields_templ = [
        fld for fld in attr_fields(inputs) if fld.metadata.get("output_file_template")
    ]
    for fld in fields_templ:
        if fld.type not in [str, ty.Union[str, bool]]:
            raise Exception(
                f"fields with output_file_template"
                " has to be a string or Union[str, bool]"
            )
        dict_[fld.name] = template_update_single(
            field=fld, inputs_dict=dict_, output_dir=output_dir
        )
    # using is and  == so it covers list and numpy arrays
    updated_templ_dict = {
        k: v
        for k, v in dict_.items()
        if not (getattr(inputs, k) is v or getattr(inputs, k) == v)
    }
    return updated_templ_dict


def template_update_single(field, inputs_dict, output_dir=None, spec_type="input"):
    """Update a single template from the input_spec or output_spec
    based on the value from inputs_dict
    (checking the types of the fields, that have "output_file_template)"
    """
    from .specs import File

    if spec_type == "input":
        if field.type not in [str, ty.Union[str, bool]]:
            raise Exception(
                f"fields with output_file_template"
                "has to be a string or Union[str, bool]"
            )
        inp_val_set = inputs_dict[field.name]
        if inp_val_set is not attr.NOTHING and not isinstance(inp_val_set, (str, bool)):
            raise Exception(
                f"{field.name} has to be str or bool, but {inp_val_set} set"
            )
        if isinstance(inp_val_set, bool) and field.type is str:
            raise Exception(
                f"type of {field.name} is str, consider using Union[str, bool]"
            )
    elif spec_type == "output":
        if field.type is not File:
            raise Exception(
                f"output {field.name} should be a File, but {field.type} set as the type"
            )
    else:
        raise Exception(f"spec_type can be input or output, but {spec_type} provided")
    # for inputs that the value is set (so the template is ignored)
    if spec_type == "input" and isinstance(inputs_dict[field.name], str):
        return inputs_dict[field.name]
    elif spec_type == "input" and inputs_dict[field.name] is False:
        # if input fld is set to False, the fld shouldn't be used (setting NOTHING)
        return attr.NOTHING
    else:  # inputs_dict[field.name] is True or spec_type is output
        template = field.metadata["output_file_template"]
        # as default, we assume that keep_extension is True
        keep_extension = field.metadata.get("keep_extension", True)
        value = _template_formatting(
            template, inputs_dict, keep_extension=keep_extension
        )
        # changing path so it is in the output_dir
        if output_dir and value is not attr.NOTHING:
            # should be converted to str, it is also used for input fields that should be str
            return str(output_dir / Path(value).name)
        else:
            return value


def _template_formatting(template, inputs_dict, keep_extension=True):
    """Formatting a single template based on values from inputs_dict.
    Taking into account that field values and template could have file extensions
    (assuming that if template has extension, the field value extension is removed,
    if field has extension, and no template extension, than it is moved to the end),
    """
    inp_fields = re.findall("{\w+}", template)
    if len(inp_fields) == 0:
        return template
    elif len(inp_fields) == 1:
        fld_name = inp_fields[0][1:-1]
        fld_value = inputs_dict[fld_name]
        if fld_value is attr.NOTHING:
            return attr.NOTHING
        fld_value_parent = Path(fld_value).parent
        fld_value_name = Path(fld_value).name

        name, *ext = fld_value_name.split(".", maxsplit=1)
        filename = str(fld_value_parent / name)

        # if keep_extension is False, the extensions are removed
        if keep_extension is False:
            ext = []
        if template.endswith(inp_fields[0]):
            # if no suffix added in template, the simplest formatting should work
            # recreating fld_value with the updated extension
            fld_value_upd = ".".join([filename] + ext)
            formatted_value = template.format(**{fld_name: fld_value_upd})
        elif "." not in template:  # the template doesn't have its own extension
            # if the fld_value has extension, it will be moved to the end
            formatted_value = ".".join([template.format(**{fld_name: filename})] + ext)
        else:  # template has its own extension
            # removing fld_value extension if any
            formatted_value = template.format(**{fld_name: filename})
        return formatted_value
    else:
        raise NotImplementedError("should we allow for more args in the template?")


def is_local_file(f):
    from .specs import File

    return f.type is File and "container_path" not in f.metadata


def is_existing_file(value):
    """ checking if an object is an existing file"""
    if isinstance(value, str) and value == "":
        return False
    try:
        return Path(value).exists()
    except TypeError:
        return False
