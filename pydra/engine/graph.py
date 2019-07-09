from copy import copy, deepcopy
from .helpers import ensure_list


class Graph:
    """
    A simple Directed Graph object

    """

    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes
        self.edges = edges
        self._create_connections()
        self._sorted_nodes = None

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        """ returns a list of all nodes or an empty list"""
        if nodes:
            nodes = ensure_list(nodes)
            if len(set(nodes)) != len(nodes):
                raise Exception("nodes have repeated elements")
            self._nodes = nodes
        else:
            self._nodes = []

    @property
    def nodes_names_map(self):
        """mapping node names to objects"""
        return {nd.name: nd for nd in self.nodes}

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, edges):
        """adding edges to the graph (nodes has to be already in self.nodes)"""
        if edges:
            edges = ensure_list(edges)
            for (nd_out, nd_in) in edges:
                if nd_out not in self.nodes or nd_in not in self.nodes:
                    raise Exception(
                        f"edge {(nd_out, nd_in)} can't be added to the graph"
                    )
            self._edges = edges
        else:
            self._edges = []

    @property
    def edges_names(self):
        """returns edges using the nodes names"""
        return [(edg[0].name, edg[1].name) for edg in self._edges]

    @property
    def sorted_nodes(self):
        """ returns sorted nodes (runs sorting if needed)"""
        if self._sorted_nodes is None:
            self.sorting()
        return self._sorted_nodes

    @property
    def sorted_nodes_names(self):
        """returns sorted nodes names"""
        return [nd.name for nd in self._sorted_nodes]

    def _create_connections(self):
        """creates connections between nodes"""
        self.connections_pred = {}
        self.connections_succ = {}
        for nd in self.nodes:
            self.connections_pred[nd.name] = []
            self.connections_succ[nd.name] = []

        for (nd_out, nd_in) in self.edges:
            self.connections_pred[nd_in.name].append(nd_out)
            self.connections_succ[nd_out.name].append(nd_in)

    def add_nodes(self, new_nodes):
        """adding new nodes and sorting the graph"""
        self.nodes = self._nodes + ensure_list(new_nodes)
        for nd in ensure_list(new_nodes):
            self.connections_pred[nd.name] = []
            self.connections_succ[nd.name] = []
        if self._sorted_nodes is not None:
            # starting from the previous sorted list, so is faster
            self.sorting(presorted=self.sorted_nodes + ensure_list(new_nodes))

    def add_edges(self, new_edges):
        """ adding new edges and sorting the graph"""
        self.edges = self._edges + ensure_list(new_edges)
        for (nd_out, nd_in) in ensure_list(new_edges):
            self.connections_pred[nd_in.name].append(nd_out)
            self.connections_succ[nd_out.name].append(nd_in)
        if self._sorted_nodes is not None:
            # starting from the previous sorted list, so it's faster
            self.sorting(presorted=self.sorted_nodes + [])

    def sorting(self, presorted=None):
        """sorting the graph, starting either from self.nodes or the previously sorted list"""
        self._sorted_nodes = []
        if presorted:
            notsorted_nodes = copy(presorted)
        else:
            notsorted_nodes = copy(self.nodes)
        connections_pred = {
            key: copy(val) for (key, val) in self.connections_pred.items()
        }

        while notsorted_nodes:
            sorted_part, notsorted_nodes = self._sorting(
                notsorted_nodes, connections_pred
            )
            self._sorted_nodes += sorted_part
            for nd_out in sorted_part:
                for nd_in in self.connections_succ[nd_out.name]:
                    connections_pred[nd_in.name].remove(nd_out)

    def _sorting(self, notsorted_list, connections_pred):
        """ adding nodes that don't have predecessors to the sorted_parts,
            returns sorted part and remaining nodes
        """
        remaining_nodes = []
        sorted_part = []
        for nd in notsorted_list:
            if not connections_pred[nd.name]:
                sorted_part.append(nd)
            else:
                remaining_nodes.append(nd)
        return sorted_part, remaining_nodes

    def remove_node(self, node):
        """removing a node, re-sorting if needed"""
        if node not in self.nodes:
            raise Exception(f"{node} is not present in the graph")
        if self.connections_pred[node.name]:
            raise Exception("this node shoudn't be run, has to wait")

        self.nodes.remove(node)
        for nd_in in self.connections_succ[node.name]:
            self.connections_pred[nd_in.name].remove(node)
            self.edges.remove((node, nd_in))
        self.connections_succ.pop(node.name)
        self.connections_pred.pop(node.name)

        # if graph is sorted, the sorted list has to be updated
        if hasattr(self, "sorted_nodes"):
            if node == self.sorted_nodes[0]:
                # if the first node is removed, no need to sort again
                self._sorted_nodes = self.sorted_nodes[1:]
            else:
                self._sorted_nodes.remove(node)
                # starting from the previous sorted list, so is faster
                self.sorting(presorted=self.sorted_nodes)

    def _checking_path(self, node_name, first_name, path=0):
        """recursive function to calculate all paths using connections list"""
        if not self.connections_succ[node_name]:
            return True
        for nd_in in self.connections_succ[node_name]:
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
        """ calculate maximum paths between any node without "history" (no predecessors)
        and all of the connections
        """
        self.max_paths = {}
        first_nodes = [key for (key, val) in self.connections_pred.items() if not val]
        for nm in first_nodes:
            self.max_paths[nm] = {}
            self._checking_path(node_name=nm, first_name=nm)
