Search.setIndex({"docnames": ["explanation/conditional-lazy", "explanation/design-approach", "explanation/hashing-caching", "explanation/provenance", "explanation/splitting-combining", "explanation/typing", "howto/create-task-package", "howto/port-from-nipype", "howto/real-example", "index", "reference/api", "tutorial/execution", "tutorial/python", "tutorial/shell", "tutorial/workflow"], "filenames": ["explanation/conditional-lazy.rst", "explanation/design-approach.rst", "explanation/hashing-caching.rst", "explanation/provenance.rst", "explanation/splitting-combining.rst", "explanation/typing.rst", "howto/create-task-package.ipynb", "howto/port-from-nipype.ipynb", "howto/real-example.ipynb", "index.rst", "reference/api.rst", "tutorial/execution.ipynb", "tutorial/python.ipynb", "tutorial/shell.ipynb", "tutorial/workflow.ipynb"], "titles": ["Conditional construction", "Design philosophy", "Caching", "Provenance", "Splitting and combining", "Type checking", "Creating a task package", "Port interfaces from Nipype", "Real-world example", "Pydra", "API", "Running tasks", "Python-task design", "Shell-task design", "Workflow design"], "terms": {"work": [0, 1, 2, 3, 4, 5, 8, 13], "progress": [0, 2, 3, 5, 8], "scientif": [1, 9], "workflow": [1, 4, 8, 9, 11, 13], "often": 1, "requir": [1, 4, 12, 13], "sophist": 1, "analys": 1, "encompass": 1, "larg": 1, "collect": [1, 14], "algorithm": 1, "The": [1, 4, 12, 13, 14], "were": 1, "origin": [1, 4], "necessarili": 1, "togeth": 1, "written": [1, 9], "differ": [1, 4], "author": 1, "some": [1, 13], "mai": 1, "python": [1, 4, 9, 11, 13, 14], "while": 1, "other": [1, 4], "might": [1, 14], "call": [1, 4, 13, 14], "extern": 1, "program": 1, "It": [1, 14], "i": [1, 4, 8, 9, 11, 12, 13, 14], "common": [1, 11], "practic": 1, "creat": [1, 4, 11, 14], "semi": 1, "manual": 1, "scientist": 1, "handl": 1, "file": [1, 11, 13, 14], "interact": 1, "partial": 1, "result": [1, 11, 12, 13, 14], "from": [1, 4, 11, 13, 14], "tool": [1, 9, 13], "thi": [1, 4, 8, 13, 14], "approach": 1, "conceptu": 1, "simpl": [1, 9, 13, 14], "easi": 1, "implement": [1, 9, 14], "time": [1, 14], "consum": 1, "error": [1, 13], "prone": 1, "difficult": 1, "share": [1, 14], "consist": [1, 11], "reproduc": 1, "scalabl": 1, "demand": 1, "organ": 1, "fulli": 1, "autom": 1, "pipelin": 1, "wa": [1, 4, 13], "motiv": 1, "behind": 1, "pydra": [1, 4, 11, 12, 13, 14], "new": [1, 9, 14], "dataflow": [1, 4, 9], "engin": [1, 4, 9, 12, 13, 14], "packag": [1, 9], "part": [1, 13], "second": [1, 4, 12, 14], "gener": [1, 9, 13, 14], "nipyp": 1, "ecosystem": 1, "an": [1, 4, 8, 9, 13, 14], "open": [1, 9, 13], "sourc": [1, 9, 13], "framework": 1, "provid": [1, 4, 9, 13], "uniform": 1, "interfac": [1, 11, 13], "exist": 1, "neuroimag": [1, 9], "softwar": [1, 9], "facilit": 1, "between": [1, 13], "compon": [1, 11, 14], "project": [1, 9], "born": 1, "commun": [1, 9], "ha": [1, 4, 9, 14], "been": [1, 14], "help": [1, 13], "build": 1, "decad": 1, "fsl": 1, "ant": 1, "afni": 1, "freesurf": 1, "spm": 1, "flexibl": [1, 4], "made": 1, "ideal": 1, "basi": 1, "popular": 1, "preprocess": 1, "fmriprep": 1, "c": [1, 11, 12, 14], "pac": 1, "meant": 1, "addit": [1, 4, 14], "being": [1, 14], "develop": [1, 9], "eas": 1, "us": [1, 4, 9, 12, 13, 14], "mind": 1, "itself": [1, 9, 14], "standalon": 1, "purpos": [1, 9], "support": [1, 4, 9], "ani": [1, 9, 12, 13, 14], "domain": [1, 9], "lightweight": [1, 9], "comput": 1, "graph": [1, 4], "construct": [1, 14], "manipul": 1, "distribut": 1, "execut": [1, 4, 9, 11, 13, 14], "well": 1, "ensur": 1, "In": [1, 4, 13, 14], "repres": [1, 4], "direct": 1, "acycl": 1, "where": [1, 4, 14], "each": [1, 4, 14], "node": [1, 4, 14], "function": [1, 9, 11, 12, 13], "anoth": 1, "reusabl": 1, "combin": [1, 11, 13], "sever": 1, "kei": [1, 4], "featur": [1, 4, 14], "make": [1, 13], "customiz": 1, "power": [1, 14], "compos": 1, "can": [1, 4, 11, 13, 14], "allow": 1, "nest": [1, 4, 13], "arbitrari": [1, 4], "depth": [1, 14], "encourag": 1, "semant": 1, "loop": [1, 4], "over": [1, 4, 14], "input": [1, 4, 13], "set": [1, 4, 14], "task": [1, 4], "run": [1, 9, 13], "paramet": [1, 4], "output": [1, 4, 11, 13, 14], "recombin": 1, "similar": [1, 4], "concept": [1, 4], "map": [1, 4, 13], "reduc": [1, 4], "model": 1, "extend": [1, 4], "A": [1, 4, 11, 14], "content": [1, 11, 13], "address": 1, "global": 1, "cach": 1, "hash": 1, "valu": [1, 4, 13, 14], "ar": [1, 4, 13, 14], "reus": 1, "previous": 1, "store": 1, "shell": [1, 9, 11, 14], "command": [1, 9, 11], "decor": [1, 13, 14], "librari": 1, "alongsid": 1, "line": 1, "integr": 1, "code": [1, 13, 14], "nativ": 1, "contain": [1, 9, 13], "associ": [1, 13], "via": [1, 9], "docker": [1, 9], "singular": [1, 9], "enabl": [1, 14], "greater": 1, "audit": 1, "proven": 1, "track": 1, "json": [1, 11], "ld": 1, "base": [1, 4], "messag": 1, "pass": [1, 13, 14], "mechan": 1, "captur": 1, "activ": 1, "These": [1, 4], "resourc": 1, "One": [4, 14], "main": 4, "goal": [4, 9], "evalu": 4, "distinguish": 4, "most": [4, 14], "complex": 4, "would": [4, 13, 14], "typic": [4, 13, 14], "involv": 4, "signific": 4, "overhead": 4, "data": 4, "manag": 4, "multipl": [4, 14], "control": 4, "specif": [4, 11, 14], "state": 4, "relat": 4, "attribut": 4, "through": 4, "method": [4, 14], "order": [4, 14], "up": 4, "done": [4, 13, 14], "": [4, 14], "simplest": 4, "exampl": [4, 13, 14], "one": [4, 13, 14], "field": [4, 14], "x": [4, 14], "therefor": [4, 9, 13], "onli": [4, 9], "wai": [4, 9, 13, 14], "its": [4, 11, 13], "assum": [4, 13, 14], "user": [4, 13], "list": [4, 13, 14], "so": 4, "copi": [4, 13], "get": [4, 14], "element": [4, 14], "follow": [4, 13], "x_1": 4, "x_2": 4, "x_n": 4, "longmapsto": 4, "also": [4, 13, 14], "diagram": 4, "1": [4, 12, 13, 14], "2": [4, 12, 13, 14], "3": [4, 11, 13, 14], "colour": 4, "stateless": 4, "after": [4, 13, 14], "runnabl": [4, 11], "whenev": 4, "more": [4, 9, 13, 14], "complic": 4, "e": [4, 9, 13], "two": [4, 11, 14], "applic": [4, 11], "thei": [4, 13, 14], "special": 4, "syntax": 4, "describ": 4, "next": 4, "perform": [4, 14], "wise": 4, "have": [4, 9, 14], "same": [4, 14], "length": 4, "tupl": [4, 12, 13, 14], "oper": 4, "parenthesi": 4, "y": [4, 14], "y_1": 4, "y_2": 4, "y_n": 4, "mapsto": 4, "option": 4, "when": [4, 13, 14], "all": [4, 11, 13, 14], "doe": 4, "squar": 4, "bracket": 4, "y_m": 4, "schemat": 4, "inp1": 4, "inp2": 4, "inp3": 4, "pairwis": 4, "merg": 4, "end": [4, 13], "need": [4, 9, 14], "explain": 4, "section": 4, "pre": 8, "process": [8, 13, 14], "t1": 8, "weight": 8, "mri": 8, "imag": [8, 13, 14], "further": [8, 14], "analysi": 8, "which": [9, 13, 14], "mix": 9, "design": 9, "see": [9, 13, 14], "philosophi": 9, "explan": 9, "pure": 9, "hand": 9, "depend": [9, 14], "straightforward": 9, "pip": 9, "Of": 9, "cours": 9, "you": [9, 14], "either": [9, 11, 13], "those": 9, "machin": 9, "g": [9, 13], "them": [9, 13], "index": 9, "modul": 9, "basic": 11, "three": 11, "type": 11, "5": [11, 13, 14], "fileformat": [11, 13, 14], "import": [11, 12, 13, 14], "loadjson": 11, "sampl": [11, 12, 13], "test": [11, 12, 13, 14], "json_fil": 11, "print": [11, 12, 13], "path": [11, 13], "refer": 11, "f": [11, 13], "name": [11, 12, 13, 14], "r": [11, 13], "read_text": [11, 13], "parameteris": [11, 13, 14], "load": 11, "load_json": 11, "out": [11, 12, 13, 14], "39": [11, 12, 13], "0uaqfzwsdk4frump48y3tt3q": 11, "34": [11, 13], "true": [11, 13], "b": [11, 12, 14], "d": [11, 12], "7": [11, 13, 14], "0": [11, 12, 13, 14], "5598136790149003": 11, "6": [11, 13, 14], "8": [12, 13, 14], "def": [12, 13, 14], "func": 12, "int": [12, 13, 14], "float": [12, 14], "return": [12, 13, 14], "samplespec": 12, "defin": [12, 13, 14], "spec": [12, 13, 14], "funcoutput": 12, "9": [12, 13, 14], "k": 12, "10": [12, 13, 14], "decim": 12, "arg": [12, 13, 14], "help_str": [12, 13, 14], "argument": [12, 13], "doubl": [12, 13], "11": [12, 13], "note": [12, 14], "we": [12, 14], "camelcas": 12, "translat": 12, "class": [12, 13, 14], "12": 12, "pprint": [12, 13], "helper": [12, 13], "fields_dict": [12, 13], "first": [12, 13, 14], "sum": [12, 14], "product": 12, "lt": [12, 13], "gt": [12, 13], "default": [12, 14], "empti": [12, 13], "convert": [12, 13, 14], "none": [12, 13, 14], "valid": [12, 13, 14], "allowed_valu": [12, 13], "xor": [12, 13], "copy_mod": [12, 13], "copymod": [12, 13], "15": [12, 13], "copy_col": [12, 13], "copycol": [12, 13], "copy_ext_decomp": [12, 13], "extensiondecomposit": [12, 13], "singl": [12, 13], "readonli": [12, 13], "fals": [12, 13], "callabl": 12, "0x10d0253a0": 12, "13": 12, "staticmethod": [12, 14], "0x10d024040": 12, "14": 12, "pythonspec": 12, "pythonoutput": 12, "0x10d024180": 12, "string": 13, "resembl": 13, "usag": 13, "quick": 13, "intuit": 13, "For": [13, 14], "cp": 13, "omit": [13, 14], "in_fil": 13, "destin": 13, "both": 13, "place": [13, 14], "within": [13, 14], "enclos": 13, "differenti": 13, "prefix": 13, "just": [13, 14], "pathlib": 13, "tempfil": 13, "mkdtemp": 13, "test_dir": 13, "test_fil": 13, "txt": 13, "w": 13, "write": 13, "cmdline": 13, "check": [13, 14], "comand": 13, "var": 13, "folder": 13, "mz": 13, "yn83q2fd3s758w1j75d2nnw80000gn": 13, "t": [13, 14], "tmpoyx19gql": 13, "If": [13, 14], "tclose": 13, "git": 13, "doc": 13, "tutori": 13, "By": 13, "consid": 13, "fsobject": 13, "howev": [13, 14], "format": [13, 14], "built": 13, "append": 13, "mime": 13, "like": [13, 14], "detail": [13, 14], "4": [13, 14], "png": [13, 14], "trimpng": 13, "trim": 13, "in_imag": 13, "out_imag": 13, "trim_png": 13, "mock": 13, "ad": [13, 14], "hyphen": 13, "immedi": 13, "space": 13, "boolean": 13, "otherwis": 13, "unless": 13, "should": [13, 14], "comma": 13, "separ": 13, "vararg": 13, "ellipsi": 13, "my_vararg": 13, "in_fs_object": 13, "object": 13, "out_dir": 13, "directori": 13, "recurs": 13, "text": 13, "text_arg": 13, "int_arg": 13, "tuple_arg": 13, "str": 13, "union": 13, "sequenc": 13, "l": 13, "dirnam": 13, "min_len": 13, "argstr": 13, "posit": 13, "sep": 13, "container_path": 13, "formatt": 13, "util": [13, 14], "multiinputobj": 13, "outarg": 13, "path_templ": 13, "keep_extens": 13, "bool": 13, "return_cod": 13, "exit": 13, "stderr": 13, "standard": 13, "stream": 13, "produc": 13, "stdout": 13, "foo": 13, "99": 13, "bar": 13, "keyword": 13, "out_fil": 13, "source_fil": 13, "entir": 13, "subtre": 13, "connect": [13, 14], "point": 13, "deriv": 13, "take": 13, "dictionari": 13, "o": [13, 14], "get_file_s": 13, "calcul": 13, "size": 13, "stat": 13, "st_size": 13, "cpwithsiz": 13, "out_file_s": 13, "cp_with_siz": 13, "256": 13, "output_dir": 13, "referenc": [13, 14], "resolv": 13, "includ": 13, "0x11481f920": 13, "To": 13, "checkabl": 13, "canon": 13, "inherit": 13, "parameter": 13, "shellspec": 13, "shelloutput": 13, "case": [13, 14], "explicitli": 13, "list_field": 13, "acommand": 13, "dag": 14, "specifi": 14, "interchang": 14, "given": 14, "add": 14, "mul": 14, "basicworkflow": 14, "correspond": 14, "outptu": 14, "downstream": 14, "placehold": 14, "statement": 14, "dure": 14, "possibl": 14, "inlin": 14, "video": 14, "shellworkflow": 14, "input_video": 14, "mp4": 14, "watermark": 14, "watermark_dim": 14, "add_watermark": 14, "ffmpeg": 14, "in_video": 14, "filter_complex": 14, "filter": 14, "out_video": 14, "overlai": 14, "output_video": 14, "handbrakecli": 14, "width": 14, "height": 14, "1280": 14, "720": 14, "implicit": 14, "detect": 14, "insid": 14, "divid": 14, "out1": 14, "out2": 14, "directaccesworkflow": 14, "demonstr": 14, "few": 14, "altern": 14, "integ": 14, "wf": 14, "z": 14, "lzout": 14, "divis": 14, "alter": 14, "initialis": 14, "directli": 14, "setoutputsofworkflow": 14, "explicit": 14, "linter": 14, "worth": 14, "extra": 14, "effort": 14, "suit": 14, "publicli": 14, "static": 14, "dataclasss": 14, "lend": 14, "custom": 14, "workflowspec": 14, "workflowoutput": 14, "a_convert": 14, "libraryworkflow": 14, "mylibraryworkflow": 14, "sometim": 14, "want": 14, "achiev": 14, "splitworkflow": 14, "multipli": 14, "sume": 14, "across": 14, "step": 14, "doesn": 14, "propag": 14, "splitthencombineworkflow": 14, "advanc": 14, "discuss": 14, "intricaci": 14, "abil": 14, "condition": 14, "conditionalworkflow": 14, "handbrake_input": 14, "els": 14, "upstream": 14, "cannot": 14, "sinc": 14, "around": 14, "limit": 14, "logic": 14, "subtract": 14, "recursivenestedworkflow": 14, "decrement_depth": 14, "out_nod": 14, "lazi": 14, "annot": 14, "strong": 14, "assign": 14, "do": 14, "conflict": 14, "typeerror": 14, "rais": 14, "best": 14, "super": 14, "expect": 14, "becaus": 14, "subtyp": 14, "jpeg": 14, "fail": 14, "clearli": 14, "intend": 14, "mp4handbrak": 14, "quicktimehandbrak": 14, "quicktim": 14, "typeerrorworkflow": 14, "ok": 14, "superclass": 14, "try": 14, "handbrak": 14, "except": 14, "now": 14, "correct": 14}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"condit": [0, 14], "construct": 0, "design": [1, 12, 13, 14], "philosophi": 1, "rational": 1, "histori": 1, "goal": 1, "cach": 2, "proven": 3, "split": [4, 14], "combin": [4, 14], "type": [4, 5, 12, 13, 14], "splitter": 4, "scalar": 4, "outer": 4, "check": [5, 12], "creat": 6, "task": [6, 10, 11, 12, 13, 14], "packag": 6, "port": 7, "interfac": 7, "from": [7, 12], "nipyp": 7, "real": 8, "world": 8, "exampl": 8, "pydra": 9, "instal": 9, "indic": 9, "tabl": 9, "api": 10, "python": [10, 12], "shell": [10, 13], "workflow": [10, 14], "specif": [10, 13], "class": 10, "run": 11, "With": 12, "augment": 12, "explicit": 12, "input": [12, 14], "output": 12, "decorated_funct": 12, "pull": 12, "help": 12, "docstr": 12, "dataclass": [12, 13, 14], "form": [12, 13, 14], "canon": 12, "work": 12, "static": 12, "command": 13, "line": 13, "templat": 13, "specifi": 13, "flag": 13, "option": 13, "default": 13, "addit": 13, "field": 13, "attribut": 13, "callabl": 13, "outptu": 13, "dynam": 13, "constructor": 14, "function": 14, "access": 14, "object": 14, "nest": 14}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "nbsphinx": 4, "sphinx.ext.viewcode": 1, "sphinx": 57}, "alltitles": {"Conditional construction": [[0, "conditional-construction"]], "Design philosophy": [[1, "design-philosophy"]], "Rationale": [[1, "rationale"]], "History": [[1, "history"]], "Goals": [[1, "goals"]], "Caching": [[2, "caching"]], "Provenance": [[3, "provenance"]], "Splitting and combining": [[4, "splitting-and-combining"]], "Types of Splitter": [[4, "types-of-splitter"]], "Scalar Splitter": [[4, "scalar-splitter"]], "Outer Splitter": [[4, "outer-splitter"]], "Type checking": [[5, "type-checking"]], "Creating a task package": [[6, "Creating-a-task-package"]], "Port interfaces from Nipype": [[7, "Port-interfaces-from-Nipype"]], "Real-world example": [[8, "Real-world-example"]], "Pydra": [[9, "pydra"]], "Installation": [[9, "installation"]], "Indices and tables": [[9, "indices-and-tables"]], "API": [[10, "api"]], "Python tasks": [[10, "python-tasks"]], "Shell tasks": [[10, "shell-tasks"]], "Workflows": [[10, "workflows"]], "Specification classes": [[10, "specification-classes"]], "Task classes": [[10, "task-classes"]], "Running tasks": [[11, "Running-tasks"]], "Python-task design": [[12, "Python-task-design"]], "With typing": [[12, "With-typing"]], "Augment with explicit inputs and outputs": [[12, "Augment-with-explicit-inputs-and-outputs"]], "Decorated_function": [[12, "Decorated_function"]], "Pull helps from docstring": [[12, "Pull-helps-from-docstring"]], "Dataclass form": [[12, "Dataclass-form"], [13, "Dataclass-form"], [14, "Dataclass-form"]], "Canonical form (to work with static type-checking)": [[12, "Canonical-form-(to-work-with-static-type-checking)"]], "Shell-task design": [[13, "Shell-task-design"]], "Command-line templates": [[13, "Command-line-templates"]], "Specifying types": [[13, "Specifying-types"]], "Flags and options": [[13, "Flags-and-options"]], "Defaults": [[13, "Defaults"]], "Additional field attributes": [[13, "Additional-field-attributes"]], "Callable outptus": [[13, "Callable-outptus"]], "Dynamic specifications": [[13, "Dynamic-specifications"]], "Workflow design": [[14, "Workflow-design"]], "Constructor functions": [[14, "Constructor-functions"]], "Accessing the workflow object": [[14, "Accessing-the-workflow-object"]], "Splitting/combining task inputs": [[14, "Splitting/combining-task-inputs"]], "Nested and conditional workflows": [[14, "Nested-and-conditional-workflows"]], "Typing": [[14, "Typing"]]}, "indexentries": {}})