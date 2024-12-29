Search.setIndex({"docnames": ["examples/t1w-preprocess", "explanation/conditional-lazy", "explanation/design-approach", "explanation/environments", "explanation/hashing-caching", "explanation/splitting-combining", "explanation/typing", "howto/create-task-package", "howto/port-from-nipype", "index", "reference/api", "tutorial/advanced-execution", "tutorial/getting-started", "tutorial/python", "tutorial/shell", "tutorial/workflow"], "filenames": ["examples/t1w-preprocess.ipynb", "explanation/conditional-lazy.rst", "explanation/design-approach.rst", "explanation/environments.rst", "explanation/hashing-caching.rst", "explanation/splitting-combining.rst", "explanation/typing.rst", "howto/create-task-package.ipynb", "howto/port-from-nipype.ipynb", "index.rst", "reference/api.rst", "tutorial/advanced-execution.ipynb", "tutorial/getting-started.ipynb", "tutorial/python.ipynb", "tutorial/shell.ipynb", "tutorial/workflow.ipynb"], "titles": ["T1w MRI preprocessing", "Conditionals and lazy fields", "Design philosophy", "Containers and environments", "Caching and hashing", "Splitting and combining", "Type checking", "Create a task package", "Port interfaces from Nipype", "Pydra", "API", "Advanced execution", "Getting started", "Python-tasks", "Shell-tasks", "Workflows"], "terms": {"thi": [0, 2, 5, 12, 14, 15], "i": [0, 2, 5, 9, 12, 13, 14, 15], "an": [0, 2, 5, 9, 12, 14, 15], "real": 0, "world": 0, "exampl": [0, 5, 12, 14, 15], "workflow": [0, 2, 5, 9, 12, 14], "pre": [0, 9, 12], "process": [0, 14, 15], "t1": 0, "weight": 0, "imag": [0, 12, 14, 15], "further": [0, 15], "analysi": 0, "work": [0, 1, 2, 3, 4, 5, 6, 12, 14], "progress": [0, 1, 3, 4, 6, 12], "scientif": [2, 9], "often": 2, "requir": [2, 5, 13, 14], "sophist": 2, "analys": 2, "encompass": 2, "larg": [2, 12], "collect": [2, 9, 15], "algorithm": 2, "The": [2, 5, 9, 12, 13, 14, 15], "were": 2, "origin": [2, 5, 12], "necessarili": 2, "togeth": [2, 12], "written": 2, "differ": [2, 5, 12], "author": 2, "some": [2, 12, 14], "mai": 2, "python": [2, 5, 9, 12, 14, 15], "while": 2, "other": [2, 5], "might": [2, 12, 15], "call": [2, 5, 12, 14, 15], "extern": 2, "program": 2, "It": [2, 12, 15], "common": [2, 12], "practic": 2, "creat": [2, 5, 9, 12, 15], "semi": 2, "manual": 2, "scientist": 2, "handl": 2, "file": [2, 12, 14, 15], "interact": 2, "partial": 2, "result": [2, 12, 13, 14, 15], "from": [2, 5, 9, 12, 14, 15], "tool": [2, 14], "approach": 2, "conceptu": [2, 12], "simpl": [2, 14, 15], "easi": 2, "implement": [2, 9, 15], "time": [2, 15], "consum": 2, "error": [2, 12, 14], "prone": 2, "difficult": 2, "share": [2, 15], "consist": 2, "reproduc": [2, 9], "scalabl": [2, 9], "demand": 2, "organ": 2, "fulli": [2, 9], "autom": [2, 9], "pipelin": 2, "wa": [2, 5, 14], "motiv": 2, "behind": 2, "pydra": [2, 5, 11, 12, 13, 14, 15], "new": [2, 15], "dataflow": [2, 5, 9], "engin": [2, 5, 9, 13, 14, 15], "packag": [2, 9, 12], "part": [2, 14], "second": [2, 5, 13, 15], "gener": [2, 9, 12, 14, 15], "nipyp": [2, 9], "ecosystem": 2, "open": [2, 12, 14], "sourc": [2, 14], "framework": 2, "provid": [2, 5, 12, 14], "uniform": 2, "interfac": [2, 9, 14], "exist": 2, "neuroimag": 2, "softwar": 2, "facilit": 2, "between": [2, 12, 14], "compon": [2, 12, 15], "project": 2, "born": 2, "commun": 2, "ha": [2, 5, 9, 15], "been": [2, 15], "help": [2, 9, 14], "build": [2, 9], "decad": 2, "fsl": [2, 9, 12], "ant": [2, 9, 12], "afni": [2, 9], "freesurf": 2, "spm": 2, "flexibl": [2, 5], "made": 2, "ideal": 2, "basi": 2, "popular": 2, "preprocess": [2, 9], "fmriprep": 2, "c": [2, 12, 13, 15], "pac": 2, "meant": 2, "addit": [2, 5, 12, 15], "being": [2, 12, 15], "develop": 2, "eas": [2, 9], "us": [2, 5, 9, 12, 13, 14, 15], "mind": 2, "itself": [2, 9, 15], "standalon": 2, "purpos": [2, 9], "support": [2, 5, 9, 11], "ani": [2, 9, 13, 14, 15], "domain": [2, 9], "lightweight": [2, 9], "comput": [2, 9], "graph": [2, 5, 9], "construct": [2, 9, 12, 15], "manipul": [2, 9], "distribut": [2, 9], "execut": [2, 5, 9, 12, 14, 15], "well": [2, 9], "ensur": 2, "In": [2, 5, 14, 15], "repres": [2, 5], "direct": 2, "acycl": 2, "where": [2, 5, 15], "each": [2, 5, 15], "node": [2, 5], "function": [2, 9, 12, 13, 14], "anoth": 2, "reusabl": [2, 9], "combin": [2, 9, 12, 14], "sever": [2, 11], "kei": [2, 5, 9], "featur": [2, 5, 9, 15], "make": [2, 12, 14], "customiz": 2, "power": [2, 9, 15], "compos": 2, "can": [2, 5, 9, 12, 14, 15], "allow": 2, "nest": [2, 5, 14], "arbitrari": [2, 5], "depth": [2, 15], "encourag": 2, "semant": [2, 9], "loop": [2, 5], "over": [2, 5, 15], "input": [2, 5, 14], "set": [2, 5, 12, 15], "task": [2, 5, 9, 11], "run": [2, 9, 14], "paramet": [2, 5, 12], "output": [2, 5, 12, 14, 15], "recombin": 2, "similar": [2, 5, 12], "concept": [2, 5], "map": [2, 5, 9, 14], "reduc": [2, 5, 9], "model": 2, "extend": [2, 5], "A": [2, 5, 15], "content": [2, 12, 14], "address": 2, "global": [2, 9], "cach": [2, 9], "hash": [2, 9, 11, 12], "valu": [2, 5, 14, 15], "ar": [2, 5, 9, 12, 14, 15], "reus": [2, 12], "previous": 2, "store": [2, 12], "shell": [2, 9, 12, 15], "command": [2, 9, 12], "decor": [2, 14, 15], "librari": 2, "alongsid": 2, "line": 2, "integr": 2, "code": [2, 12, 14, 15], "nativ": 2, "contain": [2, 9, 12, 14], "associ": [2, 14], "via": [2, 9], "docker": 2, "singular": 2, "enabl": [2, 12, 15], "greater": 2, "audit": 2, "proven": [2, 9], "track": [2, 9], "json": [2, 12], "ld": 2, "base": [2, 5], "messag": 2, "pass": [2, 12, 14, 15], "mechan": 2, "captur": 2, "activ": 2, "These": [2, 5, 9], "resourc": 2, "One": [5, 15], "main": 5, "goal": 5, "evalu": 5, "distinguish": 5, "most": [5, 15], "complex": [5, 9], "would": [5, 12, 14, 15], "typic": [5, 9, 14, 15], "involv": 5, "signific": 5, "overhead": [5, 12], "data": 5, "manag": 5, "multipl": [5, 15], "control": 5, "specif": [5, 9, 14], "state": 5, "relat": [5, 9], "attribut": 5, "through": 5, "method": [5, 12, 15], "order": [5, 15], "up": 5, "done": [5, 14, 15], "": [5, 9, 15], "simplest": 5, "one": [5, 12, 14, 15], "field": [5, 15], "x": [5, 15], "therefor": [5, 9, 12, 14], "onli": [5, 9, 12], "wai": [5, 14, 15], "its": [5, 12, 14], "assum": [5, 14, 15], "user": [5, 14], "list": [5, 12, 14, 15], "so": [5, 12], "copi": [5, 14], "get": [5, 9, 15], "element": [5, 15], "follow": [5, 14], "x_1": 5, "x_2": 5, "x_n": 5, "longmapsto": 5, "also": [5, 12, 14, 15], "diagram": 5, "1": [5, 12, 13, 14, 15], "2": [5, 12, 13, 14, 15], "3": [5, 9, 12, 14, 15], "colour": 5, "stateless": 5, "after": [5, 14, 15], "runnabl": [5, 12], "whenev": 5, "more": [5, 9, 11, 12, 14, 15], "complic": 5, "e": [5, 9, 12, 14], "two": [5, 12, 15], "applic": 5, "thei": [5, 12, 14, 15], "special": 5, "syntax": 5, "describ": 5, "next": [5, 12], "perform": [5, 12, 15], "wise": 5, "have": [5, 9, 15], "same": [5, 12, 15], "length": [5, 12], "tupl": [5, 12, 13, 14, 15], "oper": [5, 9, 12], "parenthesi": 5, "y": [5, 15], "y_1": 5, "y_2": 5, "y_n": 5, "mapsto": 5, "option": 5, "when": [5, 12, 14, 15], "all": [5, 12, 14, 15], "doe": 5, "squar": 5, "bracket": 5, "y_m": 5, "schemat": 5, "inp1": 5, "inp2": 5, "inp3": 5, "pairwis": 5, "merg": 5, "end": [5, 14], "need": [5, 9, 12, 15], "explain": 5, "section": 5, "11": [9, 13, 14], "design": [9, 13, 14, 15], "successor": 9, "http": 9, "github": 9, "com": 9, "nipi": 9, "analyt": 9, "li": 9, "creation": 9, "multiparamet": 9, "modular": [9, 12], "backend": 9, "see": [9, 11, 12, 14, 15], "advanc": [9, 15], "html": 9, "like": [9, 14, 15], "split": [9, 12], "explan": 9, "recomput": 9, "container": 9, "environ": [9, 12], "strong": [9, 15], "type": 9, "check": [9, 12, 14], "hint": 9, "philosophi": 9, "pure": 9, "which": [9, 11, 14, 15], "hand": 9, "depend": [9, 15], "straightforward": [9, 12], "pip": 9, "avail": 9, "under": [9, 12], "namespac": [9, 12], "within": [9, 12, 14, 15], "separ": [9, 12, 14], "given": [9, 15], "toolkit": 9, "g": [9, 12, 14], "niworkflow": 9, "Of": 9, "cours": 9, "you": [9, 12, 15], "either": [9, 14], "those": 9, "machin": 9, "them": [9, 12, 14], "start": 9, "t1w": 9, "mri": 9, "port": 9, "index": 9, "modul": 9, "concurrentfutur": 11, "slurm": 11, "dask": 11, "experiment": 11, "serial": 11, "debug": 11, "detail": [11, 12, 14, 15], "basic": 12, "take": [12, 14], "return": [12, 13, 14, 15], "howev": [12, 14, 15], "unlik": 12, "parameteris": [12, 14, 15], "befor": 12, "step": [12, 15], "link": 12, "worker": 12, "specifi": [12, 14, 15], "independ": 12, "encapsul": 12, "defin": [12, 13, 14, 15], "definit": [12, 15], "instal": 12, "import": [12, 13, 14, 15], "class": [12, 13, 14, 15], "instanti": 12, "object": [12, 14], "my_task": 12, "To": [12, 14], "demonstr": [12, 15], "toi": 12, "load": 12, "loadjson": 12, "we": [12, 13, 15], "test": [12, 13, 14, 15], "pathlib": [12, 14], "path": [12, 14], "tempfil": [12, 14], "mkdtemp": [12, 14], "json_cont": 12, "true": [12, 14], "b": [12, 13, 15], "d": [12, 13], "7": [12, 14, 15], "0": [12, 13, 14, 15], "55": 12, "6": [12, 14, 15], "test_dir": [12, 14], "json_fil": 12, "w": [12, 14], "f": [12, 14], "dump": 12, "now": [12, 15], "back": 12, "want": [12, 15], "load_json": 12, "access": 12, "match": 12, "assert": 12, "out": [12, 13, 14, 15], "default": [12, 13, 15], "runtim": 12, "inform": 12, "about": 12, "peak": 12, "memori": 12, "cpu": 12, "usag": [12, 14], "statu": 12, "output_dir": [12, 14], "appli": 12, "For": [12, 14, 15], "re": 12, "grid": 12, "nifti": 12, "sampl": [12, 13, 14], "ones": 12, "below": 12, "fileformat": [12, 14, 15], "medimag": 12, "nifti_dir": 12, "mkdir": 12, "rang": 12, "10": [12, 13, 14, 15], "seed": 12, "Then": 12, "mrgrid": 12, "mrtrix3": 12, "voxel": 12, "5": [12, 14, 15], "iterdir": 12, "resampl": 12, "print": [12, 13, 14], "locat": 12, "n": 12, "join": 12, "str": [12, 14], "p": 12, "possibl": [12, 15], "pair": 12, "size": [12, 14], "both": [12, 14], "splitter": 12, "must": 12, "number": 12, "voxel_s": 12, "75": 12, "25": 12, "mrgrid_varying_vox_s": 12, "cache_dir": 12, "uniqu": 12, "name": [12, 13, 14, 15], "subsequ": 12, "previou": 12, "mrgrid_varying_vox_sizes2": 12, "ident": 12, "result1": 12, "chang": 12, "result2": 12, "note": [12, 13, 15], "calcul": [12, 14], "themselv": 12, "mtime": 12, "shouldn": 12, "t": [12, 14, 15], "recalcul": 12, "unless": [12, 14], "modifi": 12, "invari": 12, "system": 12, "movement": 12, "won": 12, "invalid": 12, "renam": 12, "first_fil": 12, "with_nam": 12, "nii": 12, "gz": 12, "mrgrid_varying_vox_sizes3": 12, "result3": 12, "8": [13, 14, 15], "def": [13, 14, 15], "func": 13, "int": [13, 14, 15], "float": [13, 15], "sampledef": 13, "spec": [13, 14, 15], "funcoutput": 13, "9": [13, 14, 15], "k": 13, "decim": 13, "arg": [13, 14, 15], "help_str": [13, 14, 15], "argument": [13, 14], "doubl": [13, 14], "camelcas": 13, "translat": 13, "12": 13, "pprint": [13, 14], "helper": [13, 14], "fields_dict": [13, 14], "first": [13, 14, 15], "sum": [13, 15], "product": 13, "39": [13, 14], "lt": [13, 14], "gt": [13, 14], "empti": [13, 14], "convert": [13, 14, 15], "none": [13, 14, 15], "valid": [13, 14, 15], "allowed_valu": [13, 14], "xor": [13, 14], "copy_mod": [13, 14], "copymod": [13, 14], "15": [13, 14], "copy_col": [13, 14], "copycol": [13, 14], "copy_ext_decomp": [13, 14], "extensiondecomposit": [13, 14], "singl": [13, 14], "readonli": [13, 14], "fals": [13, 14], "callabl": 13, "0x10d0253a0": 13, "13": 13, "staticmethod": [13, 15], "0x10d024040": 13, "14": 13, "pythondef": 13, "pythonoutput": 13, "0x10d024180": 13, "string": 14, "resembl": 14, "quick": 14, "intuit": 14, "cp": 14, "omit": [14, 15], "in_fil": 14, "destin": 14, "place": [14, 15], "enclos": 14, "differenti": 14, "prefix": 14, "just": [14, 15], "test_fil": 14, "txt": 14, "write": 14, "cmdline": 14, "comand": 14, "read_text": 14, "var": 14, "folder": 14, "mz": 14, "yn83q2fd3s758w1j75d2nnw80000gn": 14, "tmpoyx19gql": 14, "If": [14, 15], "tclose": 14, "git": 14, "doc": 14, "tutori": 14, "By": 14, "consid": 14, "fsobject": 14, "format": [14, 15], "built": 14, "append": 14, "mime": 14, "4": [14, 15], "png": [14, 15], "trimpng": 14, "trim": 14, "in_imag": 14, "out_imag": 14, "trim_png": 14, "mock": 14, "ad": [14, 15], "hyphen": 14, "immedi": 14, "space": 14, "boolean": 14, "otherwis": 14, "should": [14, 15], "comma": 14, "vararg": 14, "ellipsi": 14, "my_vararg": 14, "in_fs_object": 14, "out_dir": 14, "directori": 14, "r": 14, "recurs": 14, "text": 14, "text_arg": 14, "int_arg": 14, "tuple_arg": 14, "union": 14, "sequenc": 14, "34": 14, "l": 14, "dirnam": 14, "min_len": 14, "argstr": 14, "posit": 14, "sep": 14, "container_path": 14, "formatt": 14, "util": [14, 15], "multiinputobj": 14, "outarg": 14, "path_templ": 14, "keep_extens": 14, "bool": 14, "return_cod": 14, "exit": 14, "stderr": 14, "standard": 14, "stream": 14, "produc": 14, "stdout": 14, "foo": 14, "99": 14, "bar": 14, "keyword": 14, "out_fil": 14, "source_fil": 14, "entir": 14, "subtre": 14, "connect": [14, 15], "point": 14, "deriv": 14, "dictionari": 14, "o": [14, 15], "get_file_s": 14, "stat": 14, "st_size": 14, "cpwithsiz": 14, "out_file_s": 14, "cp_with_siz": 14, "256": 14, "referenc": [14, 15], "resolv": 14, "includ": 14, "0x11481f920": 14, "checkabl": 14, "canon": 14, "inherit": 14, "parameter": 14, "shelldef": 14, "shelloutput": 14, "case": [14, 15], "explicitli": 14, "list_field": 14, "acommand": 14, "dag": 15, "interchang": 15, "add": 15, "mul": 15, "basicworkflow": 15, "correspond": 15, "outptu": 15, "downstream": 15, "placehold": 15, "statement": 15, "dure": 15, "inlin": 15, "video": 15, "shellworkflow": 15, "input_video": 15, "mp4": 15, "watermark": 15, "watermark_dim": 15, "add_watermark": 15, "ffmpeg": 15, "in_video": 15, "filter_complex": 15, "filter": 15, "out_video": 15, "overlai": 15, "output_video": 15, "handbrakecli": 15, "width": 15, "height": 15, "1280": 15, "720": 15, "implicit": 15, "detect": 15, "insid": 15, "divid": 15, "out1": 15, "out2": 15, "directaccesworkflow": 15, "few": 15, "altern": 15, "integ": 15, "wf": 15, "z": 15, "lzout": 15, "divis": 15, "alter": 15, "initialis": 15, "directli": 15, "setoutputsofworkflow": 15, "explicit": 15, "linter": 15, "worth": 15, "extra": 15, "effort": 15, "suit": 15, "publicli": 15, "static": 15, "dataclasss": 15, "lend": 15, "custom": 15, "workflowdef": 15, "workflowoutput": 15, "a_convert": 15, "libraryworkflow": 15, "mylibraryworkflow": 15, "sometim": 15, "achiev": 15, "splitworkflow": 15, "multipli": 15, "sume": 15, "across": 15, "doesn": 15, "propag": 15, "splitthencombineworkflow": 15, "discuss": 15, "intricaci": 15, "abil": 15, "condition": 15, "conditionalworkflow": 15, "handbrake_input": 15, "els": 15, "upstream": 15, "cannot": 15, "sinc": 15, "around": 15, "limit": 15, "logic": 15, "subtract": 15, "recursivenestedworkflow": 15, "decrement_depth": 15, "out_nod": 15, "lazi": 15, "annot": 15, "assign": 15, "do": 15, "conflict": 15, "typeerror": 15, "rais": 15, "best": 15, "super": 15, "expect": 15, "becaus": 15, "subtyp": 15, "jpeg": 15, "fail": 15, "clearli": 15, "intend": 15, "mp4handbrak": 15, "quicktimehandbrak": 15, "quicktim": 15, "typeerrorworkflow": 15, "ok": 15, "superclass": 15, "try": 15, "handbrak": 15, "except": 15, "correct": 15}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"t1w": 0, "mri": 0, "preprocess": 0, "condit": [1, 15], "lazi": 1, "field": [1, 14], "design": 2, "philosophi": 2, "rational": 2, "histori": 2, "goal": 2, "contain": [3, 11], "environ": [3, 11], "cach": [4, 11, 12], "hash": 4, "split": [5, 15], "combin": [5, 15], "type": [5, 6, 13, 14, 15], "splitter": 5, "scalar": 5, "outer": 5, "check": [6, 13, 15], "creat": 7, "task": [7, 10, 12, 13, 14, 15], "packag": 7, "port": 8, "interfac": 8, "from": [8, 13], "nipyp": 8, "pydra": 9, "instal": 9, "tutori": 9, "exampl": 9, "how": 9, "guid": 9, "indic": 9, "tabl": 9, "api": 10, "python": [10, 13], "shell": [10, 14], "workflow": [10, 15], "specif": 10, "class": 10, "advanc": 11, "execut": 11, "worker": 11, "result": 11, "proven": 11, "get": 12, "start": 12, "run": 12, "your": 12, "first": 12, "iter": 12, "over": 12, "input": [12, 13, 15], "directori": 12, "debug": 12, "With": 13, "augment": 13, "explicit": 13, "output": 13, "decorated_funct": 13, "pull": 13, "help": 13, "docstr": 13, "dataclass": [13, 14, 15], "form": [13, 14, 15], "canon": 13, "work": 13, "static": 13, "command": 14, "line": 14, "templat": 14, "defifi": 14, "flag": 14, "option": 14, "default": 14, "addit": 14, "attribut": 14, "callabl": 14, "outptu": 14, "dynam": 14, "definit": 14, "constructor": 15, "function": 15, "access": 15, "object": 15, "nest": 15, "between": 15, "node": 15}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "nbsphinx": 4, "sphinx.ext.viewcode": 1, "sphinx": 57}, "alltitles": {"T1w MRI preprocessing": [[0, "T1w-MRI-preprocessing"]], "Conditionals and lazy fields": [[1, "conditionals-and-lazy-fields"]], "Design philosophy": [[2, "design-philosophy"]], "Rationale": [[2, "rationale"]], "History": [[2, "history"]], "Goals": [[2, "goals"]], "Containers and environments": [[3, "containers-and-environments"]], "Caching and hashing": [[4, "caching-and-hashing"]], "Splitting and combining": [[5, "splitting-and-combining"]], "Types of Splitter": [[5, "types-of-splitter"]], "Scalar Splitter": [[5, "scalar-splitter"]], "Outer Splitter": [[5, "outer-splitter"]], "Type checking": [[6, "type-checking"]], "Create a task package": [[7, "Create-a-task-package"]], "Port interfaces from Nipype": [[8, "Port-interfaces-from-Nipype"]], "Pydra": [[9, "pydra"]], "Installation": [[9, "installation"]], "Tutorials": [[9, "tutorials"]], "Examples": [[9, "examples"]], "How-to Guides": [[9, "how-to-guides"]], "Indices and tables": [[9, "indices-and-tables"]], "API": [[10, "api"]], "Python tasks": [[10, "python-tasks"]], "Shell tasks": [[10, "shell-tasks"]], "Workflows": [[10, "workflows"], [15, "Workflows"]], "Specification classes": [[10, "specification-classes"]], "Task classes": [[10, "task-classes"]], "Advanced execution": [[11, "Advanced-execution"]], "Workers": [[11, "Workers"]], "Caching results": [[11, "Caching-results"]], "Environments (containers)": [[11, "Environments-(containers)"]], "Provenance": [[11, "Provenance"]], "Getting started": [[12, "Getting-started"]], "Running your first task": [[12, "Running-your-first-task"]], "Iterating over inputs": [[12, "Iterating-over-inputs"]], "Cache directories": [[12, "Cache-directories"]], "Debugging": [[12, "Debugging"]], "Python-tasks": [[13, "Python-tasks"]], "With typing": [[13, "With-typing"]], "Augment with explicit inputs and outputs": [[13, "Augment-with-explicit-inputs-and-outputs"]], "Decorated_function": [[13, "Decorated_function"]], "Pull helps from docstring": [[13, "Pull-helps-from-docstring"]], "Dataclass form": [[13, "Dataclass-form"], [14, "Dataclass-form"], [15, "Dataclass-form"]], "Canonical form (to work with static type-checking)": [[13, "Canonical-form-(to-work-with-static-type-checking)"]], "Shell-tasks": [[14, "Shell-tasks"]], "Command-line templates": [[14, "Command-line-templates"]], "Defifying types": [[14, "Defifying-types"]], "Flags and options": [[14, "Flags-and-options"]], "Defaults": [[14, "Defaults"]], "Additional field attributes": [[14, "Additional-field-attributes"]], "Callable outptus": [[14, "Callable-outptus"]], "Dynamic definitions": [[14, "Dynamic-definitions"]], "Constructor functions": [[15, "Constructor-functions"]], "Accessing the workflow object": [[15, "Accessing-the-workflow-object"]], "Splitting/combining task inputs": [[15, "Splitting/combining-task-inputs"]], "Nested and conditional workflows": [[15, "Nested-and-conditional-workflows"]], "Type-checking between nodes": [[15, "Type-checking-between-nodes"]]}, "indexentries": {}})