from copy import copy, deepcopy


class MyGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes
        self.edges = edges
        self._create_connections()

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        if nodes:
            if len(set(nodes)) != len(nodes):
                raise Exception("nodes have repeated elements")
            self._nodes = nodes
        else:
            self._nodes = []

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, edges):
        if edges:
            for (nd_out, nd_in) in edges:
                if nd_out not in self.nodes or nd_in not in self.nodes:
                    raise Exception(
                        f"edge {(nd_out, nd_in)} can't be added to the graph"
                    )
            self._edges = edges
        else:
            self._edges = []

    def _create_connections(self):
        self._connections_pred = {}
        self._connections_succ = {}
        for nd in self.nodes:
            self._connections_pred[nd] = []
            self._connections_succ[nd] = []

        for (nd_out, nd_in) in self.edges:
            self._connections_pred[nd_in].append(nd_out)
            self._connections_succ[nd_out].append(nd_in)

    def _sorting(self, notsorted_list, connections_pred):
        left_nodes = []
        sorted_part = []
        for nd in notsorted_list:
            if not connections_pred[nd]:
                sorted_part.append(nd)
            else:
                left_nodes.append(nd)
        return sorted_part, left_nodes

    def sorting(self, presorted=None):
        """sorting the graph"""
        self.sorted_nodes = []
        if presorted:
            notsorted_nodes = copy(presorted)
        else:
            notsorted_nodes = copy(self.nodes)
        connections_pred = deepcopy(self._connections_pred)

        while notsorted_nodes:
            sorted_part, notsorted_nodes = self._sorting(
                notsorted_nodes, connections_pred
            )
            self.sorted_nodes += sorted_part
            for nd_out in sorted_part:
                for nd_in in self._connections_succ[nd_out]:
                    connections_pred[nd_in].remove(nd_out)

    def removing(self, node):
        """removing a node, re-sorting if needed"""
        if node not in self.nodes:
            raise Exception(f"{node} is not present in the graph")
        if self._connections_pred[node]:
            raise Exception("this node shoudn't be run, has to wait")

        self.nodes.remove(node)
        for nd_in in self._connections_succ[node]:
            self._connections_pred[nd_in].remove(node)
            self.edges.remove((node, nd_in))
        self._connections_succ.pop(node)
        self._connections_pred.pop(node)

        # if graph is sorted, the sorted list has to be updated
        if hasattr(self, "sorted_nodes"):
            if node == self.sorted_nodes[0]:
                # if the first node is removed, no need to sort again
                self.sorted_nodes = self.sorted_nodes[1:]
            else:
                self.sorted_nodes.remove(node)
                # starting from the previous sorted list, so is faster
                self.sorting(presorted=self.sorted_nodes)
