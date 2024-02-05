# MIT License

# Copyright (c) 2024 Akhmed Rakhmati

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from collections.abc import Iterable
import pathlib
from typing import Optional

import graphviz
import networkx
from pyrsistent import PClass, field, pmap_field, PMap, pmap

from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.reportviews import (
    DiMultiDegreeView,
    InMultiDegreeView,
    OutMultiDegreeView,
    NodeView,
    InMultiEdgeView,
    OutMultiEdgeView,
)


class MultiDiGraph(PClass):
    _node = pmap_field(object, PMap)
    _succ = pmap_field(object, PMap)
    _pred = pmap_field(object, PMap)
    _attributes = field(PMap, initial=pmap)

    @property
    def _adj(self):
        return self._succ

    @property
    def adj(self):
        return MultiAdjacencyView(self._adj)

    @property
    def succ(self):
        return self._succ

    @property
    def pred(self):
        return self._pred

    @property
    def graph(self):
        return self._attributes

    def __len__(self):
        return len(self._node)

    def __iter__(self):
        return iter(self._node)

    def __getitem__(self, node):
        return self.adj[node]

    def __contains__(self, node):
        return node in self._node

    def __hash__(self):
        return hash((self._node, self._pred, self._succ, self._attributes))

    def is_directed(self) -> bool:
        return True

    def is_multigraph(self) -> bool:
        return True

    def add_attributes(self, **kwargs) -> "MultiDiGraph":
        _attributes = self._attributes.update(kwargs)
        return self.set(_attributes=_attributes)

    def has_node(self, node):
        return node in self

    def add_node(self, node, **kwargs):
        attributes = self._node.get(node, pmap())
        attributes = attributes.update(pmap(kwargs))
        _node = self._node.set(node, attributes)

        _pred = self._pred
        if node not in _pred:
            _pred = _pred.set(node, pmap())

        _succ = self._succ
        if node not in _succ:
            _succ = _succ.set(node, pmap())

        new_graph = self.set(_node=_node, _pred=_pred, _succ=_succ)
        return new_graph

    def add_nodes_from(self, nodes_for_adding, **new_attributes):
        new_graph = self
        for node in nodes_for_adding:
            if isinstance(node, tuple):
                node, ndict = node
                newdict = pmap(new_attributes).update(ndict)
            else:
                newdict = pmap(new_attributes)
            attributes = self._node[node] if node in self._node else pmap()
            attributes = attributes.update(newdict)
            new_graph = new_graph.add_node(node, **attributes)
        return new_graph

    def add_edge(self, source, sink, key=None, **kwargs):
        if source not in self:
            self = self.add_node(source)
        if sink not in self:
            self = self.add_node(sink)

        _node = self._node

        def _add_edge(node_to_neighbors, from_node, to_node, edge_key):
            neighbors = node_to_neighbors.get(from_node, pmap())
            edges = neighbors.get(to_node, pmap())
            if edge_key is None:
                edge_key = max(edges.keys()) + 1 if edges.keys() else 0
            edges = edges.set(edge_key, pmap(kwargs))
            neighbors = neighbors.set(to_node, edges)
            node_to_neighbors = node_to_neighbors.set(from_node, neighbors)
            return node_to_neighbors

        _pred = _add_edge(self._pred, sink, source, key)
        _succ = _add_edge(self._succ, source, sink, key)

        new_graph = self.set(_node=_node, _pred=_pred, _succ=_succ)
        return new_graph

    def add_edges_from(self, ebunch_to_add, **attr):
        new_graph = self
        for edge in ebunch_to_add:
            if len(edge) == 4:
                source, sink, key, data = edge
            elif len(edge) == 3:
                source, sink, data = edge
                key = None
            elif len(edge) == 2:
                source, sink = edge
                key = None
                data = {}
            else:
                raise networkx.NetworkXError(
                    f"Edge tuple {edge} must be a 2-tuple or 3-tuple."
                )
            new_graph = new_graph.add_edge(source, sink, key, **data, **attr)
        return new_graph

    def remove_node(self, node):
        for source, sink in self.edges(node):
            self = self.remove_edge(source, sink)

        _node = self._node.remove(node)
        _succ = self._succ
        _pred = self._pred

        new_graph = self.set(_node=_node, _pred=_pred, _succ=_succ)
        return new_graph

    def remove_edge(self, source, sink, key=None):
        def _remove_edge(node_to_neighbors, from_node, to_node, edge_key):
            neighbors = node_to_neighbors.get(from_node, pmap())
            edges = neighbors.get(to_node, pmap())
            if not edges:
                raise networkx.NetworkXError("There is no edge to remove!")
            elif edge_key is None:
                edge_key = max(edges.keys())
            elif edge_key not in edges:
                raise networkx.NetworkXError("There is no edge to remove!")
            edges = edges.remove(edge_key)
            neighbors = neighbors.set(to_node, edges)
            node_to_neighbors = node_to_neighbors.set(from_node, neighbors)
            return node_to_neighbors

        _pred = _remove_edge(self._pred, sink, source, key)
        _succ = _remove_edge(self._succ, source, sink, key)

        new_graph = self.set(_pred=_pred, _succ=_succ)
        return new_graph

    @property
    def nodes(self, data=False, default=None):
        return NodeView(self)(data=data, default=default)

    def edges(self, nbunch=None, data=False, keys=False, default=None):
        if nbunch is not None and not isinstance(nbunch, Iterable):
            if nbunch not in self:
                raise KeyError(f"{nbunch} is not a node in the graph")
        return OutMultiEdgeView(self)(
            data=data, nbunch=nbunch, keys=keys, default=default
        )

    out_edges = edges

    def in_edges(self, nbunch=None, data=False, keys=False, default=None):
        if nbunch is not None and not isinstance(nbunch, Iterable):
            if nbunch not in self:
                raise KeyError(f"{nbunch} is not a node in the graph")
        return InMultiEdgeView(self)(
            data=data, nbunch=nbunch, keys=keys, default=default
        )

    def nbunch_iter(self, nbunch=None):
        if nbunch is None:
            return iter(self)
        if not isinstance(nbunch, Iterable):
            nbunch = [nbunch]
        return (node for node in nbunch if node in self)

    def neighbors(self, node):
        return iter(self.adj[node])

    def degree(self, nbunch=None, weight=None):
        return DiMultiDegreeView(self)(nbunch, weight)

    def in_degree(self, nbunch=None, weight=None):
        return InMultiDegreeView(self)(nbunch, weight)

    def out_degree(self, nbunch=None, weight=None):
        return OutMultiDegreeView(self)(nbunch, weight)

    size = networkx.MultiDiGraph.size
    number_of_edges = networkx.MultiDiGraph.number_of_edges

    def has_successor(self, node, successor):
        return successor in self._succ[node]

    def successors(self, node):
        return iter(self._succ[node].keys())

    def has_predecessor(self, node, predecessor):
        return predecessor in self._pred[node]

    def predecessors(self, node):
        return iter(self.pred[node].keys())

    def to_undirected(self, **kwargs) -> "networkx.MultiGraph":
        return to_networkx(self).to_undirected(**kwargs)

    def reverse(self, **kwargs) -> "networkx.MultiDiGraph":
        graph = to_networkx(self)
        graph = graph.reverse(**kwargs)
        graph = from_networkx(graph)
        return graph

    def subgraph(self, nodes) -> "networkx.MultiDiGraph":
        graph = to_networkx(self)
        graph = graph.subgraph(nodes)
        graph = from_networkx(graph)
        return graph


def topological_traversal(graph):
    return networkx.topological_sort(graph)


def default_visualize_node(graphviz_graph, graph, node):
    graphviz_graph.node(node.name, label=f"{node}")


def default_visualize_edge(graphviz_graph, graph, edge):
    source, sink, keys, data = edge
    graphviz_graph.edge(
        source.name,
        sink.name,
    )


def visualize_graph(
    graph: MultiDiGraph,
    *,
    graphviz_graph=None,
    visualize_node=default_visualize_node,
    visualize_edge=default_visualize_edge,
    file_name: Optional[pathlib.Path] = None,
) -> None:
    if graphviz_graph is None:
        graphviz_graph = graphviz.Digraph()

    for node in graph:
        visualize_node(graphviz_graph, graph, node)

    for node in graph:
        for edge in graph.in_edges(node, data=True, keys=True):
            visualize_edge(graphviz_graph, graph, edge)

    if file_name is not None:
        if not isinstance(file_name, pathlib.Path):
            raise TypeError(f"file_name must be a pathlib.Path, not {type(file_name)}")
        if file_name.suffix != ".svg":
            raise ValueError(
                f"file_name must have a .svg suffix, not {file_name.suffix}"
            )
        format = file_name.suffix[1:]
        graphviz_graph.render(file_name.with_suffix(""), format=format)

    return graphviz_graph


def from_networkx(nx_graph: networkx.MultiDiGraph) -> MultiDiGraph:
    graph = MultiDiGraph()

    for node, data in nx_graph.nodes(data=True):
        graph = graph.add_node(node, **data)

    for source, sink, key, data in nx_graph.edges(keys=True, data=True):
        graph = graph.add_edge(source, sink, key, **data)

    graph = graph.add_attributes(**nx_graph.graph)

    return graph


def to_networkx(graph) -> "networkx.MultiDiGraph":
    nx_graph = networkx.MultiDiGraph()

    for node, data in graph.nodes(data=True):
        nx_graph.add_node(node, **data)

    for source, sink, key, data in graph.edges(keys=True, data=True):
        nx_graph.add_edge(source, sink, key, **data)

    nx_graph.graph.update(**graph.graph)
    return nx_graph


def compose_all(*graphs) -> "MultiDiGraph":
    new_graph, *_ = tuple(graphs)

    for graph in graphs:
        new_graph = new_graph.add_nodes_from(graph.nodes(data=True))

    for graph in graphs:
        new_graph = new_graph.add_edges_from(graph.edges(keys=True, data=True))

    return new_graph


def merge_graphs(*graph_node_pairs) -> "MultiDiGraph":
    def merge_binary(graph_a, graph_b: "MultiDiGraph", graph_b_node):
        graph_b_operands = list(graph_b.predecessors(graph_b_node))
        if graph_b_operands and all(operand in graph_a for operand in graph_b_operands):
            _node = graph_a._node.set(graph_b_node, graph_b._node[graph_b_node])
            new_graph = graph_a.set(_node=_node)
            for source, sink, key, data in graph_b.in_edges(
                graph_b_node, keys=True, data=True
            ):
                new_graph = new_graph.add_edge(source, sink, key, **data)
            return new_graph

        else:
            _node = graph_a._node.update(graph_b._node)

            def merge_edges(node_to_neighbors, graph_b_node_to_neighbors):
                for (
                    graph_b_node,
                    graph_b_neighbors,
                ) in graph_b_node_to_neighbors.items():
                    if graph_b_node not in node_to_neighbors:
                        node_to_neighbors = node_to_neighbors.set(
                            graph_b_node, graph_b_neighbors
                        )
                        continue

                    neighbors = node_to_neighbors[graph_b_node]
                    for graph_b_neighbor, graph_b_edges in graph_b_neighbors.items():
                        if graph_b_neighbor not in neighbors:
                            neighbors = neighbors.set(graph_b_neighbor, graph_b_edges)
                            continue

                        edges = neighbors[graph_b_neighbor].update(
                            graph_b_neighbors[graph_b_neighbor]
                        )
                        neighbors = neighbors.set(graph_b_neighbor, edges)

                    node_to_neighbors = node_to_neighbors.set(graph_b_node, neighbors)
                return node_to_neighbors

            _pred = merge_edges(graph_a._pred, graph_b._pred)
            _succ = merge_edges(graph_a._succ, graph_b._succ)

            new_graph = graph_a.set(_node=_node, _pred=_pred, _succ=_succ)
            return new_graph

    (graph, node), *graph_node_pairs = graph_node_pairs
    for other_graph, other_node in graph_node_pairs:
        graph = merge_binary(graph, other_graph, other_node)
    return graph
