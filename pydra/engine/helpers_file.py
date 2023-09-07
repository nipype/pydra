"""Functions ported from Nipype 1, after removing parts that were related to py2."""

import os
import re
import logging
from pathlib import Path
import typing as ty
from copy import copy
import subprocess as sp
from contextlib import contextmanager
import attr
from fileformats.core import FileSet


logger = logging.getLogger("pydra")


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


def copy_nested_files(
    value: ty.Any,
    dest_dir: os.PathLike,
    supported_modes: FileSet.CopyMode = FileSet.CopyMode.any,
    **kwargs,
) -> ty.Any:
    """Copies all "file-sets" found within the nested value (e.g. dict, list,...) into the
    destination directory. If no nested file-sets are found then the original value is
    returned. Note that multiple nested file-sets (e.g. a list) will to have unique names
    names (i.e. not differentiated by parent directories) otherwise there will be a path
    clash in the destination directory.

    Parameters
    ----------
    value : Any
        the value to copy files from (if required)
    dest_dir : os.PathLike
        the destination directory to copy the files to
    **kwargs
        passed directly onto FileSet.copy()
    """
    from ..utils.typing import TypeParser  # noqa

    cache: ty.Dict[FileSet, FileSet] = {}

    def copy_fileset(fileset: FileSet):
        try:
            return cache[fileset]
        except KeyError:
            pass
        supported = supported_modes
        if any(MountIndentifier.on_cifs(p) for p in fileset.fspaths):
            supported -= FileSet.CopyMode.symlink
        if not all(
            MountIndentifier.on_same_mount(p, dest_dir) for p in fileset.fspaths
        ):
            supported -= FileSet.CopyMode.hardlink
        copied = fileset.copy(dest_dir=dest_dir, supported_modes=supported, **kwargs)
        cache[fileset] = copied
        return copied

    return TypeParser.apply_to_instances(FileSet, copy_fileset, value)


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

    # Collect templated inputs for which all requirements are satisfied.
    fields_templ = [
        field
        for field in attr_fields(inputs)
        if field.metadata.get("output_file_template")
        and getattr(inputs, field.name) is not False
        and all(
            getattr(inputs, required_field) is not attr.NOTHING
            for required_field in field.metadata.get("requires", ())
        )
    ]

    dict_mod = {}
    for fld in fields_templ:
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
    # if input_dict_st with state specific value is not available,
    # the dictionary will be created from inputs object
    from ..utils.typing import TypeParser  # noqa
    from pydra.engine.specs import LazyField, OUTPUT_TEMPLATE_TYPES

    if inputs_dict_st is None:
        inputs_dict_st = attr.asdict(inputs, recurse=False)

    if spec_type == "input":
        inp_val_set = inputs_dict_st[field.name]
        if isinstance(inp_val_set, bool) and field.type in (Path, str):
            raise TypeError(
                f"type of '{field.name}' is Path, consider using Union[Path, bool]"
            )
        if inp_val_set is not attr.NOTHING and not isinstance(inp_val_set, LazyField):
            inp_val_set = TypeParser(ty.Union[OUTPUT_TEMPLATE_TYPES])(inp_val_set)
    elif spec_type == "output":
        if not TypeParser.contains_type(FileSet, field.type):
            raise TypeError(
                f"output {field.name} should be file-system object, but {field.type} "
                "set as the type"
            )
    else:
        raise TypeError(f"spec_type can be input or output, but {spec_type} provided")
    # for inputs that the value is set (so the template is ignored)
    if spec_type == "input":
        if isinstance(inp_val_set, (Path, list)):
            return inp_val_set
        if inp_val_set is False:
            # if input fld is set to False, the fld shouldn't be used (setting NOTHING)
            return attr.NOTHING
    # inputs_dict[field.name] is True or spec_type is output
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
    # if a template is a function it has to be run first with the inputs as the only arg
    template = field.metadata["output_file_template"]
    if callable(template):
        template = template(inputs)

    # as default, we assume that keep_extension is True
    if isinstance(template, (tuple, list)):
        formatted = [
            _string_template_formatting(field, t, inputs, inputs_dict_st)
            for t in template
        ]
    else:
        assert isinstance(template, str)
        formatted = _string_template_formatting(field, template, inputs, inputs_dict_st)
    return formatted


def _string_template_formatting(field, template, inputs, inputs_dict_st):
    from .specs import MultiInputObj, MultiOutputFile

    keep_extension = field.metadata.get("keep_extension", True)
    inp_fields = re.findall(r"{\w+}", template)
    inp_fields_fl = re.findall(r"{\w+:[0-9.]+f}", template)
    inp_fields += [re.sub(":[0-9.]+f", "", el) for el in inp_fields_fl]
    if len(inp_fields) == 0:
        return template

    val_dict = {}
    file_template = None

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
            if isinstance(fld_value, os.PathLike) or (
                isinstance(fld_value, str) and "." in fld_value
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
        [isinstance(el, (list, MultiInputObj)) for el in val_dict.values()]
    ):
        # all fields that are lists
        keys_list = [
            k for k, el in val_dict.items() if isinstance(el, (list, MultiInputObj))
        ]
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
    from ..utils.typing import TypeParser

    return "container_path" not in f.metadata and TypeParser.contains_type(
        FileSet, f.type
    )


class MountIndentifier:
    """Used to check the mount type that given file paths reside on in order to determine
    features that can be used (e.g. symlinks)"""

    @classmethod
    def on_cifs(cls, path: os.PathLike) -> bool:
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

        NB: This function and sub-functions are copied from the nipype.utils.filemanip module


        NB: Adapted from https://github.com/nipy/nipype
        """
        return cls.get_mount(path)[1] == "cifs"

    @classmethod
    def on_same_mount(cls, path1: os.PathLike, path2: os.PathLike) -> bool:
        """Checks whether two or paths are on the same logical file system"""
        return cls.get_mount(path1)[0] == cls.get_mount(path2)[0]

    @classmethod
    def get_mount(cls, path: os.PathLike) -> ty.Tuple[Path, str]:
        """Get the mount point for a given file-system path

        Parameters
        ----------
        path: os.PathLike
            the file-system path to identify the mount of

        Returns
        -------
        mount_point: os.PathLike
            the root of the mount the path sits on
        fstype : str
            the type of the file-system (e.g. ext4 or cifs)"""
        try:
            # Only the first match (most recent parent) counts, mount table sorted longest
            # to shortest
            return next(
                (Path(p), t)
                for p, t in cls.get_mount_table()
                if str(path).startswith(p)
            )
        except StopIteration:
            return (Path("/"), "ext4")

    @classmethod
    def generate_cifs_table(cls) -> ty.List[ty.Tuple[str, str]]:
        """
        Construct a reverse-length-ordered list of mount points that fall under a CIFS mount.

        This precomputation allows efficient checking for whether a given path
        would be on a CIFS filesystem.
        On systems without a ``mount`` command, or with no CIFS mounts, returns an
        empty list.

        """
        exit_code, output = sp.getstatusoutput("mount")
        return cls.parse_mount_table(exit_code, output)

    @classmethod
    def parse_mount_table(
        cls, exit_code: int, output: str
    ) -> ty.List[ty.Tuple[str, str]]:
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

    @classmethod
    def get_mount_table(cls) -> ty.List[ty.Tuple[str, str]]:
        if cls._mount_table is None:
            cls._mount_table = cls.generate_cifs_table()
        return cls._mount_table

    @classmethod
    @contextmanager
    def patch_table(cls, mount_table: ty.List[ty.Tuple[str, str]]):
        """Patch the mount table with new values. Used in test routines"""
        orig_table = cls._mount_table
        cls._mount_table = list(mount_table)
        try:
            yield
        finally:
            cls._mount_table = orig_table

    _mount_table: ty.Optional[ty.List[ty.Tuple[str, str]]] = None
