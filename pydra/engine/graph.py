"""Data structure to support :class:`~pydra.engine.core.Workflow` tasks."""
from copy import copy
from .helpers import ensure_list


class DiGraph:
    """A simple Directed Graph object."""

    def __init__(self, nodes=None, edges=None):
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
        self._nodes = []
        self.nodes = nodes
        self._edges = []
        self.edges = edges
        self._create_connections()
        self._sorted_nodes = None
        self._node_wip = []

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

    def remove_nodes(self, nodes):
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

        """
        nodes = ensure_list(nodes)
        for nd in nodes:
            if nd not in self.nodes:
                raise Exception(f"{nd} is not present in the graph")
            if self.predecessors[nd.name]:
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
