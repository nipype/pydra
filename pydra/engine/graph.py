"""Data structure to support :class:`~pydra.engine.core.Workflow` tasks."""
from copy import copy
from pathlib import Path
import subprocess as sp

from .helpers import ensure_list


class DiGraph:
    """A simple Directed Graph object."""

    def __init__(self, name=None, nodes=None, edges=None):
        """
        Initialize a directed graph.

        Parameters
        ----------
        nodes :
            Tasks are represented by the nodes of the graph.
        edges :
            Connections of inputs and outputs between tasks in
            the graph.

        """
        self.name = name
        self._nodes = []
        self.nodes = nodes
        self._edges = []
        self.edges = edges
        self._create_connections()
        self._sorted_nodes = None
        self._node_wip = []
        self._nodes_details = {}

    def copy(self):
        """
        Duplicate this graph.

        Create a copy that contains new lists and dictionaries,
        but runnable objects are the same.

        """
        cls = self.__class__
        new_graph = cls.__new__(cls)
        new_graph._nodes = self._nodes[:]
        new_graph._node_wip = self._node_wip[:]
        new_graph._edges = self._edges[:]
        if self._sorted_nodes:
            new_graph._sorted_nodes = self._sorted_nodes[:]
        else:
            new_graph._sorted_nodes = None
        new_graph.predecessors = {}
        for key, val in self.predecessors.items():
            new_graph.predecessors[key] = self.predecessors[key][:]
        new_graph.successors = {}
        for key, val in self.successors.items():
            new_graph.successors[key] = self.successors[key][:]
        return new_graph

    @property
    def nodes(self):
        """Get a list of the nodes currently contained in the graph."""
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        if nodes:
            nodes = ensure_list(nodes)
            if len(set(nodes)) != len(nodes):
                raise Exception("nodes have repeated elements")
            self._nodes = nodes

    @property
    def nodes_names_map(self):
        """Get a map of node names to nodes."""
        return {nd.name: nd for nd in self.nodes}

    @property
    def edges(self):
        """Get a list of the links between nodes."""
        return self._edges

    @edges.setter
    def edges(self, edges):
        """Add edges to the graph (nodes should be already set)."""
        if edges:
            edges = ensure_list(edges)
            for (nd_out, nd_in) in edges:
                if nd_out not in self.nodes or nd_in not in self.nodes:
                    raise Exception(
                        f"edge {(nd_out, nd_in)} can't be added to the graph"
                    )
            self._edges = edges

    @property
    def edges_names(self):
        """Get edges as pairs of the nodes they connect."""
        return [(edg[0].name, edg[1].name) for edg in self._edges]

    @property
    def nodes_details(self):
        """ dictionary with details of the nodes
            for each task, there are inputs/outputs and connections
            (with input/output fields names)
        """
        # removing repeated fields from inputs and outputs
        for el in self._nodes_details.values():
            el["inputs"] = list(set(el["inputs"]))
            el["inputs"].sort()
            el["outputs"] = list(set(el["outputs"]))
            el["outputs"].sort()
        return self._nodes_details

    @property
    def sorted_nodes(self):
        """Return sorted nodes (runs sorting if needed)."""
        if self._sorted_nodes is None:
            self.sorting()
        return self._sorted_nodes

    @property
    def sorted_nodes_names(self):
        """Return a list of sorted nodes names."""
        return [nd.name for nd in self._sorted_nodes]

    def _create_connections(self):
        """Create connections between nodes."""
        self.predecessors = {}
        self.successors = {}
        for nd in self.nodes:
            self.predecessors[nd.name] = []
            self.successors[nd.name] = []

        for (nd_out, nd_in) in self.edges:
            self.predecessors[nd_in.name].append(nd_out)
            self.successors[nd_out.name].append(nd_in)

    def add_nodes(self, new_nodes):
        """Insert new nodes and sort the new graph."""
        self.nodes = self._nodes + ensure_list(new_nodes)
        for nd in ensure_list(new_nodes):
            self.predecessors[nd.name] = []
            self.successors[nd.name] = []
        if self._sorted_nodes is not None:
            # starting from the previous sorted list, so is faster
            self.sorting(presorted=self.sorted_nodes + ensure_list(new_nodes))

    def add_edges(self, new_edges):
        """Add new edges and sort the new graph."""
        self.edges = self._edges + ensure_list(new_edges)
        for (nd_out, nd_in) in ensure_list(new_edges):
            self.predecessors[nd_in.name].append(nd_out)
            self.successors[nd_out.name].append(nd_in)
        if self._sorted_nodes is not None:
            # starting from the previous sorted list, so it's faster
            self.sorting(presorted=self.sorted_nodes + [])

    def add_edges_description(self, new_edge_details):
        """ adding detailed description of the connections, filling _nodes_details"""
        in_nd, in_fld, out_nd, out_fld = new_edge_details
        for key in [in_nd, out_nd]:
            self._nodes_details.setdefault(
                key, {"inputs": [], "outputs": [], "connections": []}
            )

        if (in_fld, out_nd, out_fld) not in self._nodes_details[in_nd]["connections"]:
            self._nodes_details[in_nd]["connections"].append((in_fld, out_nd, out_fld))
            self._nodes_details[in_nd]["inputs"].append(in_fld)
            self._nodes_details[out_nd]["outputs"].append(out_fld)

    def sorting(self, presorted=None):
        """
        Sort this graph.

        Sorting starts either from self.nodes or the
        previously sorted list.

        Parameters
        ----------
        presorted : :obj:`list`
            A list of previously sorted nodes.

        """
        self._sorted_nodes = []
        if presorted:
            notsorted_nodes = copy(presorted)
        else:
            notsorted_nodes = copy(self.nodes)
        predecessors = {key: copy(val) for (key, val) in self.predecessors.items()}

        # nodes that depends only on the self._nodes_wip should go first
        # soe remove them from the connections
        for nd_out in self._node_wip:
            for nd_in in self.successors[nd_out.name]:
                predecessors[nd_in.name].remove(nd_out)

        while notsorted_nodes:
            sorted_part, notsorted_nodes = self._sorting(notsorted_nodes, predecessors)
            self._sorted_nodes += sorted_part
            for nd_out in sorted_part:
                for nd_in in self.successors[nd_out.name]:
                    predecessors[nd_in.name].remove(nd_out)

    def _sorting(self, notsorted_list, predecessors):
        """
        Sort implementation.

        Adding nodes that don't have predecessors to the sorted_parts,
        returns sorted part and remaining nodes.

        """
        remaining_nodes = []
        sorted_part = []
        for nd in notsorted_list:
            if not predecessors[nd.name]:
                sorted_part.append(nd)
            else:
                remaining_nodes.append(nd)
        return sorted_part, remaining_nodes

    def remove_nodes(self, nodes, check_ready=True):
        """
        Mark nodes for removal from the graph, re-sorting if needed.

        .. important ::
            This method does not remove connections, see
            :py:meth:`~DiGraph.remove_node_connections`.
            Nodes are added to the ``_node_wip`` list, marking
            them for removal when all referring connections
            are removed.

        Parameters
        ----------
        nodes : :obj:`list`
            List of nodes to be marked for removal.
        check_ready : :obj: `bool`
            checking if the node is ready to be removed

        """
        nodes = ensure_list(nodes)
        for nd in nodes:
            if nd not in self.nodes:
                raise Exception(f"{nd} is not present in the graph")
            if self.predecessors[nd.name] and check_ready:
                raise Exception("this node shoudn't be run, has to wait")
            self.nodes.remove(nd)
            # adding the node to self._node_wip as for
            self._node_wip.append(nd)
        # if graph is sorted, the sorted list has to be updated
        if hasattr(self, "sorted_nodes"):
            if nodes == self.sorted_nodes[: len(nodes)]:
                # if the first node is removed, no need to sort again
                self._sorted_nodes = self.sorted_nodes[len(nodes) :]
            else:
                for nd in nodes:
                    self._sorted_nodes.remove(nd)
                # starting from the previous sorted list, so is faster
                self.sorting(presorted=self.sorted_nodes)

    def remove_nodes_connections(self, nodes):
        """
        Remove connections between nodes.

        Also prunes the nodes from ``_node_wip``.

        Parameters
        ----------
        nodes : :obj:`list`
            List of nodes which connections are to be removed.

        """
        nodes = ensure_list(nodes)
        for nd in nodes:
            for nd_in in self.successors[nd.name]:
                self.predecessors[nd_in.name].remove(nd)
                self.edges.remove((nd, nd_in))
            self.successors.pop(nd.name)
            self.predecessors.pop(nd.name)
            self._node_wip.remove(nd)

    def remove_previous_connections(self, nodes):
        """
        Remove connections that the node has with predecessors.

        Also prunes the nodes from ``_node_wip``.

        Parameters
        ----------
        nodes : :obj:`list`
            List of nodes which connections are to be removed.

        """
        nodes = ensure_list(nodes)
        for nd in nodes:
            for nd_out in self.predecessors[nd.name]:
                if nd_out.name in self.successors:
                    self.successors[nd_out.name].remove(nd)
                self.edges.remove((nd_out, nd))
            self.successors.pop(nd.name)
            self.predecessors.pop(nd.name)
            self._node_wip.remove(nd)

    def _checking_successors_nodes(self, node, remove=True):
        if self.successors[node.name]:
            for nd_in in self.successors[node.name]:
                self._successors_all.append(nd_in)
                self._checking_successors_nodes(node=nd_in)
        else:
            return True

    def remove_successors_nodes(self, node):
        """ Removing all the nodes that follow the node"""
        self._successors_all = []
        self._checking_successors_nodes(node=node, remove=False)
        self.remove_nodes_connections(nodes=node)
        nodes_removed = []
        for nd in self._successors_all:
            if nd in self.nodes:
                nodes_removed.append(nd.name)
                self.remove_nodes(nodes=nd, check_ready=False)
                self.remove_previous_connections(nodes=nd)
        return set(nodes_removed)

    def _checking_path(self, node_name, first_name, path=0):
        """Calculate all paths using connections list (re-entering function)."""
        if not self.successors[node_name]:
            return True
        for nd_in in self.successors[node_name]:
            if nd_in.name in self.max_paths[first_name].keys():
                # chose the maximum paths
                self.max_paths[first_name][nd_in.name] = max(
                    self.max_paths[first_name][nd_in.name], path + 1
                )
            else:
                self.max_paths[first_name][nd_in.name] = path + 1
            self._checking_path(
                node_name=nd_in.name, first_name=first_name, path=path + 1
            )

    def calculate_max_paths(self):
        """
        Calculate maximum paths.

        Maximum paths are calculated between any node without "history" (no predecessors)
        and all of the connections.

        """
        self.max_paths = {}
        first_nodes = [key for (key, val) in self.predecessors.items() if not val]
        for nm in first_nodes:
            self.max_paths[nm] = {}
            self._checking_path(node_name=nm, first_name=nm)

    def create_dotfile_simple(self, outdir, name="graph"):
        """ creates a simple dotfile (no nested structure)"""
        from .core import is_workflow

        dotstr = "digraph G {\n"
        for nd in self.nodes:
            if is_workflow(nd):
                if nd.state:
                    # adding color for wf with a state
                    dotstr += f"{nd.name} [shape=box, color=blue]\n"
                else:
                    dotstr += f"{nd.name} [shape=box]\n"
            else:
                if nd.state:
                    # adding color for nd with a state
                    dotstr += f"{nd.name} [color=blue]\n"
                else:
                    dotstr += f"{nd.name}\n"
        for ed in self.edges:
            # if the tails nd has a final state, than the state is passed
            # to the next node and the arrow is blue
            if ed[0].state and ed[0].state.splitter_rpn_final:
                dotstr += f"{ed[0].name} -> {ed[1].name} [color=blue]\n"
            else:
                dotstr += f"{ed[0].name} -> {ed[1].name}\n"

        dotstr += "}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        dotfile = Path(outdir) / f"{name}.dot"
        dotfile.write_text(dotstr)
        return dotfile

    def create_dotfile_detailed(self, outdir, name="graph_det"):
        """ creates a detailed dotfile (detailed connections - input/output fields,
        but no nested structure)
        """
        dotstr = "digraph structs {\n"
        dotstr += "node [shape=record];\n"
        if not self._nodes_details:
            raise Exception("node_details is empty, detailed dotfile can't be created")
        for nd_nm, nd_det in self.nodes_details.items():
            if nd_nm == self.name:  # the main workflow itself
                # wf inputs
                wf_inputs_str = f'{{<{nd_det["outputs"][0]}> {nd_det["outputs"][0]}'
                for el in nd_det["outputs"][1:]:
                    wf_inputs_str += f" | <{el}> {el}"
                wf_inputs_str += "}"
                dotstr += f'struct_{nd_nm} [color=red, label="{{WORKFLOW INPUT: | {wf_inputs_str}}}"];\n'
                # wf outputs
                wf_outputs_str = f'{{<{nd_det["inputs"][0]}> {nd_det["inputs"][0]}'
                for el in nd_det["inputs"][1:]:
                    wf_outputs_str += f" | <{el}> {el}"
                wf_outputs_str += "}"
                dotstr += f'struct_{nd_nm}_out [color=red, label="{{WORKFLOW OUTPUT: | {wf_outputs_str}}}"];\n'
                # connections to the wf outputs
                for con in nd_det["connections"]:
                    dotstr += (
                        f"struct_{con[1]}:{con[2]} -> struct_{nd_nm}_out:{con[0]};\n"
                    )
            else:  # elements of the main workflow
                inputs_str = "{INPUT:"
                for inp in nd_det["inputs"]:
                    inputs_str += f" | <{inp}> {inp}"
                inputs_str += "}"
                outputs_str = "{OUTPUT:"
                for out in nd_det["outputs"]:
                    outputs_str += f" | <{out}> {out}"
                outputs_str += "}"
                dotstr += f'struct_{nd_nm} [shape=record, label="{inputs_str} | {nd_nm} | {outputs_str}"];\n'
                # connections between elements
                for con in nd_det["connections"]:
                    dotstr += f"struct_{con[1]}:{con[2]} -> struct_{nd_nm}:{con[0]};\n"
        dotstr += "}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        dotfile = Path(outdir) / f"{name}.dot"
        dotfile.write_text(dotstr)
        return dotfile

    def create_dotfile_nested(self, outdir, name="graph"):
        """dotfile that includes the nested structures for workflows"""
        dotstr = "digraph G {\ncompound=true \n"
        dotstr += self._create_dotfile_single_graph(nodes=self.nodes, edges=self.edges)
        dotstr += "}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        dotfile = Path(outdir) / f"{name}.dot"
        dotfile.write_text(dotstr)
        return dotfile

    def _create_dotfile_single_graph(self, nodes, edges):
        from .core import is_workflow

        wf_asnd = []
        dotstr = ""
        for nd in nodes:
            if is_workflow(nd):
                wf_asnd.append(nd.name)
                for task in nd.graph.nodes:
                    nd.create_connections(task)
                dotstr += f"subgraph cluster_{nd.name} {{\n" f"label = {nd.name} \n"
                dotstr += self._create_dotfile_single_graph(
                    nodes=nd.graph.nodes, edges=nd.graph.edges
                )
                if nd.state:
                    dotstr += "color=blue\n"
                dotstr += "}\n"
            else:
                if nd.state:
                    dotstr += f"{nd.name} [color=blue]\n"
                else:
                    dotstr += f"{nd.name}\n"

        dotstr_edg = ""
        for ed in edges:
            if ed[0].name in wf_asnd and ed[1].name in wf_asnd:
                head_nd = list(ed[1].nodes)[0].name
                tail_nd = list(ed[0].nodes)[-1].name
                dotstr_edg += (
                    f"{tail_nd} -> {head_nd} "
                    f"[ltail=cluster_{ed[0].name}, "
                    f"lhead=cluster_{ed[1].name}]\n"
                )
            elif ed[0].name in wf_asnd:
                tail_nd = list(ed[0].nodes)[-1].name
                dotstr_edg += (
                    f"{tail_nd} -> {ed[1].name} [ltail=cluster_{ed[0].name}]\n"
                )
            elif ed[1].name in wf_asnd:
                head_nd = list(ed[1].nodes)[0].name
                dotstr_edg += (
                    f"{ed[0].name} -> {head_nd} [lhead=cluster_{ed[1].name}]\n"
                )
            else:
                dotstr_edg += f"{ed[0].name} -> {ed[1].name}\n"
            # if tail has a state and a final splitter (not combined) than the state
            # is passed to the next state and the arrow is blue
            if ed[0].state and ed[0].state.splitter_rpn_final:
                dotstr_edg = dotstr_edg[:-1] + " [color=blue]\n"
        dotstr = dotstr + dotstr_edg
        return dotstr

    def export_graph(self, dotfile, ext="png"):
        """ exporting dotfile to other format, equires the dot command"""
        available_ext = [
            "bmp",
            "canon",
            "cgimage",
            "cmap",
            "cmapx",
            "cmapx_np",
            "dot",
            "dot_json",
            "eps",
            "exr",
            "fig",
            "gif",
            "gv",
            "icns",
            "ico",
            "imap",
            "imap_np",
            "ismap",
            "jp2",
            "jpe",
            "jpeg",
            "jpg",
            "json",
            "json0",
            "mp",
            "pct",
            "pdf",
            "pic",
            "pict",
            "plain",
            "plain-ext",
            "png",
            "pov",
            "ps",
            "ps2",
            "psd",
            "sgi",
            "svg",
            "svgz",
            "tga",
            "tif",
            "tiff",
            "tk",
            "vml",
            "vmlz",
            "xdot",
            "xdot1.2",
            "xdot1.4",
            "xdot_json",
        ]
        if ext not in available_ext:
            raise Exception(f"unvalid extension - {ext}, chose from {available_ext}")

        dot_check = sp.run(["which", "dot"], stdout=sp.PIPE, stderr=sp.PIPE)
        if not dot_check.stdout:
            raise Exception(f"dot command not available, can't create a {ext} file")

        formatted_dot = dotfile.with_suffix(f".{ext}")
        cmd = f"dot -T{ext} -o {formatted_dot} {dotfile}"
        sp.run(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
        return formatted_dot
