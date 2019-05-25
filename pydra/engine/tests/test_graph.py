from ..graph import Graph
import pytest


class ObjTest:
    def __init__(self, name):
        self.name = name


A = ObjTest("a")
B = ObjTest("b")
C = ObjTest("c")
D = ObjTest("d")
E = ObjTest("e")


def test_no_edges():
    """a, b"""
    graph = Graph(nodes=[A, B])

    # checking nodes and edges
    assert [nd.name for nd in graph.nodes] == ["a", "b"]
    assert [(edg[0].name, edg[1].name) for edg in graph.edges] == []
    # checking names
    assert graph.nodes_names == ["a", "b"]
    assert graph.edges_names == []


def test_edges_1():
    """a -> b"""
    graph = Graph(nodes=[A, B], edges=[(A, B)])

    assert [nd.name for nd in graph.nodes] == ["a", "b"]
    assert [(edg[0].name, edg[1].name) for edg in graph.edges] == [("a", "b")]

    assert graph.nodes_names == ["a", "b"]
    assert graph.edges_names == [("a", "b")]


def test_edges_1a():
    """a -> b (add_nodes and add_edges)"""
    graph = Graph()
    graph.add_nodes([A, B])
    graph.add_edges((A, B))

    assert graph.nodes_names == ["a", "b"]
    assert graph.edges_names == [("a", "b")]


def test_edges_2():
    """a -> b"""
    graph = Graph(nodes=[B, A], edges=[(A, B)])
    assert graph.nodes_names == ["b", "a"]
    assert graph.edges_names == [("a", "b")]


def test_edges_3():
    """a-> b -> c; a -> c; d"""
    graph = Graph(nodes=[B, A, C, D], edges=[(A, B), (B, C), (A, C)])
    assert graph.nodes_names == ["b", "a", "c", "d"]
    assert graph.edges_names == [("a", "b"), ("b", "c"), ("a", "c")]


def test_edges_ecxeption_1():
    with pytest.raises(Exception) as excinfo:
        graph = Graph(nodes=[A, B, A], edges=[(A, B)])
    assert "repeated elements" in str(excinfo.value)


def test_edges_ecxeption_2():
    with pytest.raises(Exception) as excinfo:
        graph = Graph(nodes=[A, B], edges=[(A, C)])
    assert "can't be added" in str(excinfo.value)


def test_sort_1():
    """a -> b"""
    graph = Graph(nodes=[A, B], edges=[(A, B)])
    assert graph.nodes_names == ["a", "b"]
    assert graph.edges_names == [("a", "b")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "b"]


def test_sort_2():
    """a -> b"""
    graph = Graph(nodes=[B, A], edges=[(A, B)])
    assert graph.nodes_names == ["b", "a"]
    assert graph.edges_names == [("a", "b")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "b"]


def test_sort_3():
    """a-> b -> c; a -> c; d"""
    graph = Graph(nodes=[B, A, C, D], edges=[(A, B), (B, C), (A, C)])
    assert graph.nodes_names == ["b", "a", "c", "d"]
    assert graph.edges_names == [("a", "b"), ("b", "c"), ("a", "c")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "d", "b", "c"]


def test_sort_4():
    """a-> b -> c; a -> c; a -> d"""
    graph = Graph(nodes=[B, A, C, D], edges=[(A, B), (B, C), (A, C), (A, D)])
    assert graph.nodes_names == ["b", "a", "c", "d"]
    assert graph.edges_names == [("a", "b"), ("b", "c"), ("a", "c"), ("a", "d")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "b", "d", "c"]


def test_sort_5():
    """a-> b -> c; a -> c; d -> c"""
    graph = Graph(nodes=[B, A, C, D], edges=[(A, B), (B, C), (A, C), (D, C)])
    assert graph.nodes_names == ["b", "a", "c", "d"]
    assert graph.edges_names == [("a", "b"), ("b", "c"), ("a", "c"), ("d", "c")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "d", "b", "c"]


def test_sort_5a():
    """a-> b -> c; a -> c; d -> c (add_nodes/edges)"""
    graph = Graph(nodes=[A, C, D], edges=[(A, C), (D, C)])
    graph.add_nodes(B)
    graph.add_edges([(A, B), (B, C)])
    assert graph.nodes_names == ["a", "c", "d", "b"]
    assert graph.edges_names == [("a", "c"), ("d", "c"), ("a", "b"), ("b", "c")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "d", "b", "c"]


def test_sort_5b():
    """a-> b -> c; a -> c; d -> c (add_nodes/edges)"""
    graph = Graph(nodes=[A, C, D], edges=[(A, C), (D, C)])
    assert graph.nodes_names == ["a", "c", "d"]
    assert graph.edges_names == [("a", "c"), ("d", "c")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "d", "c"]

    graph.add_nodes(B)
    assert graph.nodes_names == ["a", "c", "d", "b"]
    assert graph.edges_names == [("a", "c"), ("d", "c")]
    assert graph.sorted_nodes_names == ["a", "d", "b", "c"]

    graph.add_edges([(A, B), (B, C)])
    assert graph.nodes_names == ["a", "c", "d", "b"]
    assert graph.edges_names == [("a", "c"), ("d", "c"), ("a", "b"), ("b", "c")]
    assert graph.sorted_nodes_names == ["a", "d", "b", "c"]


def test_sort_6():
    """a -> b -> c -> e; a -> c -> e; a -> b -> d -> e"""
    graph = Graph(
        nodes=[D, E, C, B, A], edges=[(A, B), (B, C), (A, C), (B, D), (C, E), (D, E)]
    )
    assert graph.nodes_names == ["d", "e", "c", "b", "a"]
    assert graph.edges_names == [
        ("a", "b"),
        ("b", "c"),
        ("a", "c"),
        ("b", "d"),
        ("c", "e"),
        ("d", "e"),
    ]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "b", "d", "c", "e"]


def test_remove_1():
    """a -> b (removing a node)"""
    graph = Graph(nodes=[A, B], edges=[(A, B)])
    assert graph.nodes_names == ["a", "b"]
    assert graph.edges_names == [("a", "b")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "b"]

    # removing a node (e.g. after is sent to run)
    graph.remove_node(A)
    assert graph.nodes_names == ["b"]
    assert graph.edges_names == []
    assert graph.sorted_nodes_names == ["b"]


def test_remove_2():
    """a-> b -> c; a -> c; d (removing a node)"""
    graph = Graph(nodes=[B, A, C, D], edges=[(A, B), (B, C), (A, C)])
    assert graph.nodes_names == ["b", "a", "c", "d"]
    assert graph.edges_names == [("a", "b"), ("b", "c"), ("a", "c")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "d", "b", "c"]

    graph.remove_node(A)
    assert graph.nodes_names == ["b", "c", "d"]
    assert graph.edges_names == [("b", "c")]
    assert graph.sorted_nodes_names == ["d", "b", "c"]


def test_remove_3():
    """a-> b -> c; a -> c; d (removing a node)"""
    graph = Graph(nodes=[B, A, C, D], edges=[(A, B), (B, C), (A, C)])
    assert graph.nodes_names == ["b", "a", "c", "d"]
    assert graph.edges_names == [("a", "b"), ("b", "c"), ("a", "c")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "d", "b", "c"]

    graph.remove_node(D)
    assert graph.nodes_names == ["b", "a", "c"]
    assert graph.edges_names == [("a", "b"), ("b", "c"), ("a", "c")]
    assert graph.sorted_nodes_names == ["a", "b", "c"]


def test_remove_4():
    """ a-> b -> c; a -> d -> e (removing a node)"""
    graph = Graph(nodes=[B, A, C, D, E], edges=[(A, B), (B, C), (A, D), (D, E)])
    assert graph.nodes_names == ["b", "a", "c", "d", "e"]
    assert graph.edges_names == [("a", "b"), ("b", "c"), ("a", "d"), ("d", "e")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "b", "d", "c", "e"]

    graph.remove_node(A)
    assert graph.nodes_names == ["b", "c", "d", "e"]
    assert graph.edges_names == [("b", "c"), ("d", "e")]
    assert graph.sorted_nodes_names == ["b", "d", "c", "e"]

    graph.remove_node(D)
    assert graph.nodes_names == ["b", "c", "e"]
    assert graph.edges_names == [("b", "c")]
    assert graph.sorted_nodes_names == ["b", "e", "c"]


def test_remove_exception_1():
    """a -> b (removing a node)"""
    graph = Graph(nodes=[A, B], edges=[(A, B)])
    assert graph.nodes_names == ["a", "b"]
    assert graph.edges_names == [("a", "b")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "b"]

    with pytest.raises(Exception) as excinfo:
        graph.remove_node(B)
    assert "has to wait" in str(excinfo.value)


def test_remove_add_1():
    """a -> b (removing and adding nodes)"""
    graph = Graph(nodes=[A, B], edges=[(A, B)])
    assert graph.nodes_names == ["a", "b"]
    assert graph.edges_names == [("a", "b")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "b"]

    # removing a node (e.g. after is sent to run)
    graph.remove_node(A)
    assert graph.nodes_names == ["b"]
    assert graph.edges_names == []
    assert graph.sorted_nodes_names == ["b"]

    graph.add_nodes(A)
    graph.add_edges((A, B))
    assert graph.nodes_names == ["b", "a"]
    assert graph.edges_names == [("a", "b")]
    assert graph.sorted_nodes_names == ["a", "b"]


def test_remove_add_2():
    """a-> b -> c; a -> c; d (removing and adding nodes)"""
    graph = Graph(nodes=[B, A, C, D], edges=[(A, B), (B, C), (A, C)])
    assert graph.nodes_names == ["b", "a", "c", "d"]
    assert graph.edges_names == [("a", "b"), ("b", "c"), ("a", "c")]

    graph.sorting()
    assert graph.sorted_nodes_names == ["a", "d", "b", "c"]

    graph.remove_node(A)
    assert graph.nodes_names == ["b", "c", "d"]
    assert graph.edges_names == [("b", "c")]
    assert graph.sorted_nodes_names == ["d", "b", "c"]

    graph.add_nodes(A)
    graph.add_edges([(A, B), (A, C)])
    assert graph.nodes_names == ["b", "c", "d", "a"]
    assert graph.edges_names == [("b", "c"), ("a", "b"), ("a", "c")]
    assert graph.sorted_nodes_names == ["d", "a", "b", "c"]
