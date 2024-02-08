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

from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import inspect
import math
from loguru import logger
import pathlib
import time
from typing import Any, Callable, Optional, Union, Tuple

import graphviz
import torch
from pyrsistent import PClass, field

from torchtrail import multidigraph


TORCH_NN_MODULE_CALL = torch.nn.Module.__call__


# The following functions are overriden to capture input tensors
TORCH_CREATION_OPERATION_NAMES = [
    "as_tensor",
    "from_numpy",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "arange",
    "range",
    "linspace",
    "logspace",
    "eye",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "complex",
    "heaviside",
    "bernoulli",
    "multinomial",
    "normal",
    "poisson",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
]
TORCH_CREATION_OPERATIONS = [
    getattr(torch, name) for name in TORCH_CREATION_OPERATION_NAMES
]


class Node(PClass):
    name = field(type=str, mandatory=True)

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name


class TorchTensor(PClass):

    def __repr__(self):
        return "torch.Tensor"


class TorchParameter(PClass):
    parameter = field(mandatory=True)

    def __repr__(self):
        output = "torch.nn.Parameter"
        if hasattr(self.parameter, "torchtrail_name"):
            output = f"{output}\nname: {self.parameter.torchtrail_name}"
        return output


class TorchFunction(PClass):
    function = field(mandatory=True)

    def __repr__(self):
        return self.function.__name__


class TorchModule(PClass):
    module = field(mandatory=True)
    graph: multidigraph.MultiDiGraph = field(mandatory=True)
    inputs: list[TracedTensor] = field(mandatory=True)
    outputs: list[TracedTensor] = field(mandatory=True)

    def __repr__(self):
        output = type(self.module).__name__
        if self.module.torchtrail_name != "":
            return f"{output}\nname: {self.module.torchtrail_name}"
        return output


class TorchModuleInput(PClass):
    name: str = field(mandatory=True)

    def __repr__(self):
        return f"TorchModuleInput\nname: {self.name}"


class LazyTensor:
    def __init__(self, graph: multidigraph.MultiDiGraph, node: Node, output_index: int):
        self.graph: multidigraph.MultiDiGraph = graph
        self.node: Node = node
        self.output_index: int = output_index

    @classmethod
    def from_traced_tensor(cls, traced_tesnor: TracedTorchTensor):
        return cls(traced_tesnor.graph, traced_tesnor.node, traced_tesnor.output_index)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def shape(self) -> str:
        return self.graph.nodes[self.node]["shapes"][self.output_index]


UNIQUE_ID = 0


def get_unique_id():
    global UNIQUE_ID
    output = UNIQUE_ID
    UNIQUE_ID += 1
    return output


def create_input_tensor(
    tensor: torch.Tensor, function: Optional[Callable[..., Any]] = None
) -> TracedTorchTensor:
    if isinstance(tensor, torch.nn.Parameter):
        node_name = f"torch_parameter_{get_unique_id()}"
        node = Node(name=node_name)
        graph = multidigraph.MultiDiGraph().add_node(
            node,
            operation=TorchParameter(parameter=tensor),
            shapes=(tuple(tensor.shape),),
            dtypes=(tensor.dtype,),
        )
        return TracedTorchTensor(tensor, graph=graph, node=node, output_index=0)
    elif isinstance(tensor, torch.Tensor):
        if function is None:
            operation = TorchTensor()
        else:
            operation = TorchFunction(function=function)
        node_name = f"torch_input_{get_unique_id()}"
        node = Node(name=node_name)
        graph = multidigraph.MultiDiGraph().add_node(
            node,
            operation=operation,
            shapes=(tuple(tensor.shape),),
            dtypes=(tensor.dtype,),
        )
        return TracedTorchTensor(tensor, graph=graph, node=node, output_index=0)
    else:
        raise RuntimeError(f"Unknown input type: {type(tensor)}")


def preprocess_args_and_kwargs(*args, **kwargs) -> Any:
    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, TracedTorchTensor):
            return arg
        elif isinstance(arg, torch.Tensor):
            return create_input_tensor(arg)
        else:
            return arg

    args = [preprocess_arg(arg) for arg in args]
    kwargs = {name: preprocess_arg(arg) for name, arg in kwargs.items()}
    return args, kwargs


def get_input_tensors(object):
    input_tensors = []
    if isinstance(object, TracedTensor):
        input_tensors.append(object)
    elif isinstance(object, (list, tuple)):
        for element in object:
            input_tensors += get_input_tensors(element)
    elif isinstance(object, dict):
        for value in object.values():
            input_tensors += get_input_tensors(value)
    return input_tensors


def preprocess_return_value(return_value):
    output_tensors = []
    if isinstance(return_value, (int, torch.Size, torch.device, torch.dtype, str)):
        pass
    elif isinstance(return_value, torch.Tensor) and not isinstance(
        return_value, TracedTensor
    ):
        raise RuntimeError(f"Expected TracedTorchTensor but got torch.Tensor")
    elif isinstance(return_value, TracedTorchTensor):
        output_tensors.append(return_value)
    elif isinstance(return_value, (tuple, list)):
        for value in return_value:
            output_tensors += preprocess_return_value(value)
    elif dataclasses.is_dataclass(return_value):
        for class_field in dataclasses.fields(return_value):
            value = getattr(return_value, class_field.name)
            output_tensors += preprocess_return_value(value)
    elif isinstance(return_value, dict):
        for value in return_value.values():
            output_tensors += preprocess_return_value(value)
    elif return_value is None:
        pass
    else:
        raise RuntimeError(f"Unexpected type {type(return_value)}")
    return output_tensors


def postprocess_return_value(return_value, output_tensors):
    if isinstance(return_value, (int, torch.Size, torch.device, torch.dtype, str)):
        return return_value
    elif isinstance(return_value, torch.Tensor) and not isinstance(
        return_value, TracedTensor
    ):
        raise RuntimeError(f"Expected TracedTensor but got torch.Tensor")
    elif isinstance(return_value, TracedTensor):
        output_tensor, *_ = output_tensors
        output_tensors.pop(0)
        return output_tensor
    elif isinstance(return_value, tuple):
        return tuple(
            postprocess_return_value(value, output_tensors) for value in return_value
        )
    elif dataclasses.is_dataclass(return_value):
        updated_fields = {}
        for class_field in dataclasses.fields(return_value):
            value = getattr(return_value, class_field.name)
            updated_fields[class_field.name] = postprocess_return_value(
                value, output_tensors
            )
        return type(return_value)(**updated_fields)
    elif isinstance(return_value, dict):
        return {
            name: postprocess_return_value(value, output_tensors)
            for name, value in return_value.items()
        }
    else:
        return return_value


class TracedTensor: ...


class TracedTorchTensor(torch.Tensor, TracedTensor):
    @staticmethod
    def __new__(
        cls: Any,
        tensor: Any,
        graph: multidigraph.MultiDiGraph,
        node: Node,
        output_index: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return super().__new__(cls, tensor, *args, **kwargs)  # type: ignore[call-arg]

    def __init__(
        self,
        tensor: Any,
        *,
        graph: multidigraph.MultiDiGraph,
        node: Node,
        output_index: int,
    ):
        self.graph: multidigraph.MultiDiGraph = graph
        self.node: Node = node
        self.output_index: int = output_index

    @property
    def name(self) -> str:
        return self.node.name

    @classmethod
    def __torch_function__(
        cls: Any,
        function,
        types: Any,
        args: Any = (),
        kwargs: Any = None,
    ) -> Any:

        if kwargs is None:
            kwargs = {}

        args, kwargs = preprocess_args_and_kwargs(*args, **kwargs)
        input_tensors = get_input_tensors(args) + get_input_tensors(kwargs)

        start_time = time.time()
        function_return_value = super().__torch_function__(
            function, types, args, kwargs
        )
        end_time = time.time()
        duration = end_time - start_time

        output_tensors = preprocess_return_value(function_return_value)
        if not output_tensors:
            return function_return_value

        shapes = tuple(tuple(tensor.shape) for tensor in output_tensors)
        dtypes = tuple(tensor.dtype for tensor in output_tensors)

        node_name = f"{function.__name__}_{get_unique_id()}"
        node = Node(name=node_name)
        graph = multidigraph.merge_graphs(
            *((tensor.graph, tensor.node) for tensor in input_tensors)
        )
        graph = graph.add_node(
            node,
            operation=TorchFunction(function=function),
            shapes=shapes,
            dtypes=dtypes,
            duration=duration,
        )
        for input_index, tensor in enumerate(input_tensors):
            graph = graph.add_edge(
                tensor.node,
                node,
                source_output_index=tensor.output_index,
                sink_input_index=input_index,
            )

        output_tensors = [
            TracedTorchTensor(tensor, graph=graph, node=node, output_index=output_index)
            for output_index, tensor in enumerate(output_tensors)
        ]
        return postprocess_return_value(function_return_value, output_tensors)


def wrap_create_function(function: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> TracedTensor:
        input_tensor = function(*args, **kwargs)
        return create_input_tensor(input_tensor, function)

    return wrapper


def create_module_input(name, tensor: torch.Tensor) -> TracedTensor:
    node_name = f"module_input_{get_unique_id()}"
    node = Node(name=node_name)
    graph = multidigraph.MultiDiGraph().add_node(
        node,
        operation=TorchModuleInput(name=name),
        shapes=(tuple(tensor.shape),),
        dtypes=(tensor.dtype,),
    )
    return TracedTorchTensor(tensor, graph=graph, node=node, output_index=0)


def convert_to_module_args_and_kwargs(signature, *args, **kwargs) -> Any:

    def preprocess_arg(name: str, arg: Any) -> Any:
        if isinstance(arg, TracedTensor):
            return create_module_input(name, arg)
        elif isinstance(arg, torch.nn.Parameter):
            raise RuntimeError("Module parameters are not supported")
        else:
            return arg

    arg_names = signature.args[1:]
    index = 0
    while len(arg_names) < len(args):
        arg_names.append(f"arg_{index}")
        index += 1
    args = [preprocess_arg(name, arg) for name, arg in zip(arg_names, args)]
    kwargs = {name: preprocess_arg(name, arg) for name, arg in kwargs.items()}
    return args, kwargs


def create_module(module, module_input_tensors, module_output_tensors):
    module_inputs = [
        LazyTensor.from_traced_tensor(tensor) for tensor in module_input_tensors
    ]
    module_outputs = [
        LazyTensor.from_traced_tensor(tensor) for tensor in module_output_tensors
    ]
    module_graph = multidigraph.compose_all(
        *[tensor.graph for tensor in module_inputs + module_outputs]
    )
    operation = TorchModule(
        module=module,
        graph=module_graph,
        inputs=module_inputs,
        outputs=module_outputs,
    )
    return operation


def traced_module_forward(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:

    for name, child in module.named_modules():
        if not hasattr(child, "torchtrail_name"):
            child.torchtrail_name = name

    for name, parameter in module.named_parameters():
        if not hasattr(parameter, "torchtrail_name"):
            parameter.torchtrail_name = name

    args, kwargs = preprocess_args_and_kwargs(*args, **kwargs)

    module_args, module_kwargs = convert_to_module_args_and_kwargs(
        inspect.getfullargspec(module.forward), *args, **kwargs
    )

    start_time = time.time()
    module_return_value = TORCH_NN_MODULE_CALL(module, *module_args, **module_kwargs)
    end_time = time.time()
    duration = end_time - start_time

    module_input_tensors = get_input_tensors(module_args) + get_input_tensors(
        module_kwargs
    )
    module_output_tensors = preprocess_return_value(module_return_value)
    input_tensors = get_input_tensors(args) + get_input_tensors(kwargs)
    graph = multidigraph.merge_graphs(
        *((tensor.graph, tensor.node) for tensor in input_tensors)
    )

    shapes = tuple(tuple(tensor.shape) for tensor in module_output_tensors)
    dtypes = tuple(tensor.dtype for tensor in module_output_tensors)

    operation = create_module(module, module_input_tensors, module_output_tensors)
    node_name = f"{module.torchtrail_name}_{get_unique_id()}"
    node = Node(name=node_name)

    graph = graph.add_node(
        node,
        operation=operation,
        shapes=shapes,
        dtypes=dtypes,
        duration=duration,
    )

    for input_index, tensor in enumerate(input_tensors):
        graph = graph.add_edge(
            tensor.node,
            node,
            source_output_index=tensor.output_index,
            sink_input_index=input_index,
        )

    output_tensors = [
        TracedTorchTensor(tensor, graph=graph, node=node, output_index=output_index)
        for output_index, tensor in enumerate(module_output_tensors)
    ]
    return postprocess_return_value(module_return_value, output_tensors)


@contextmanager
def trace():
    try:

        # Monkey-patch module __call__ and torch creation ops
        setattr(torch.nn.Module, "__call__", traced_module_forward)

        for name, op in zip(TORCH_CREATION_OPERATION_NAMES, TORCH_CREATION_OPERATIONS):
            setattr(torch, name, wrap_create_function(op))

        yield

    finally:
        # Reset monkey-patched module __call__ and torch creation ops
        setattr(torch.nn.Module, "__call__", TORCH_NN_MODULE_CALL)

        for name, op in zip(TORCH_CREATION_OPERATION_NAMES, TORCH_CREATION_OPERATIONS):
            setattr(torch, name, op)


LEVEL_COLORS = [
    "#0000ff80",
    "#ee00ee80   ",
    "#ff000080",
    "#eeee0080",
    "#00ff0080",
    "#00eeee80",
]


def get_source(graph, node, source_output_index, level, max_depth=None):
    operation = graph.nodes[node]["operation"]

    if max_depth is not None and level + 1 == max_depth:
        return node

    if not isinstance(operation, TorchModule):
        return node

    module = operation
    module_graph = module.graph
    module_node = module.outputs[source_output_index].node
    return get_source(
        module_graph,
        module_node,
        source_output_index,
        level=level + 1,
        max_depth=max_depth,
    )


def get_sink(graph, node, sink_input_index, level, max_depth=None):
    operation = graph.nodes[node]["operation"]

    if not isinstance(operation, TorchModule):
        return node

    if max_depth is not None and level + 1 == max_depth:
        return node

    module = operation
    module_graph = module.graph
    module_node = module.inputs[sink_input_index].node
    return get_sink(
        module_graph,
        module_node,
        sink_input_index,
        level=level + 1,
        max_depth=max_depth,
    )


def visualize_node(
    graphviz_graph, graph, node, max_depth, visualize_node, visualize_edge, level
):
    attributes = graph.nodes[node]
    operation = attributes["operation"]
    name = node.name

    input_tensors = []
    for source, _, edge_data in graph.in_edges(node, data=True):
        input_shape = graph.nodes[source]["shapes"][edge_data["source_output_index"]]
        input_dtype = graph.nodes[source]["dtypes"][edge_data["source_output_index"]]
        input_tensors.append((input_shape, input_dtype))

    output_shapes = attributes["shapes"]
    output_dtypes = attributes["dtypes"]
    output_tensors = tuple(
        (shape, dtype) for shape, dtype in zip(output_shapes, output_dtypes)
    )
    label = f"{operation}"

    duration = attributes.get("duration", None)
    if duration is not None:
        label = f"{label}\nduration: {duration:.9f}s"

    num_columns = max(len(input_tensors), len(output_tensors))

    table = label.replace("\n", "<BR/>")
    table = f"""<
            <TABLE BORDER="{0}" CELLBORDER="{1}"
            CELLSPACING="{1}" CELLPADDING="{1}">
            <TR>
                <TD ROWSPAN="{2 if input_tensors else 1}">{table}</TD>
            """

    def compute_even_column_sizes(num_columns, num_tensors):
        if num_tensors == 0:
            return []
        column_size = math.ceil(num_columns // num_tensors)
        column_sizes = []
        remaining = num_columns
        for _ in range(num_tensors):
            if remaining > column_size:
                column_sizes.append(column_size)
            else:
                column_sizes.append(remaining)
            remaining -= column_size
        return column_sizes

    input_column_sizes = compute_even_column_sizes(num_columns, len(input_tensors))
    for index, (shape, dtype) in enumerate(input_tensors):
        column_size = input_column_sizes[index]
        table = (
            table
            + f"""
                <TD COLSPAN="{column_size}">Input: {index}<BR/>{shape}<BR/>{dtype}</TD>
            """
        )
    table += """
            </TR>
            <TR>
            """
    output_column_sizes = compute_even_column_sizes(num_columns, len(output_tensors))
    for index, (shape, dtype) in enumerate(output_tensors):
        column_size = output_column_sizes[index]
        table += f"""
                <TD COLSPAN="{column_size}">Output: {index}<BR/>{shape}<BR/>{dtype}</TD>
            """
    table += f"""
            </TR>
            </TABLE>>"""

    if isinstance(operation, TorchModule):
        color = LEVEL_COLORS[level % len(LEVEL_COLORS)]
        if max_depth is None or level < max_depth - 1:
            with graphviz_graph.subgraph(name=node.name) as cluster_graph:
                cluster_graph.attr(
                    fontcolor="black",
                    bgcolor=color,
                    cluster="true",
                    label=label,
                    rankdir="TB",
                    shape="hexagon",
                )
                cluster_graph.node_attr["style"] = "filled"
                _visualize(
                    cluster_graph,
                    operation.graph,
                    max_depth=max_depth,
                    level=level + 1,
                    visualize_node=visualize_node,
                    visualize_edge=visualize_edge,
                )
        else:
            graphviz_graph.node(
                name,
                label=table,
                fontcolor="black",
                fillcolor=color,
            )

    else:
        graphviz_graph.node(
            name,
            label=table,
            fillcolor="#DCDCDC",
        )


def visualize_edge(graphviz_graph, graph, edge, max_depth, level):
    source, sink, _, edge_data = edge

    source_output_index = edge_data["source_output_index"]
    sink_input_index = edge_data["sink_input_index"]

    source_name = get_source(
        graph,
        source,
        source_output_index,
        level,
        max_depth=max_depth,
    ).name

    sink_name = get_sink(
        graph,
        sink,
        sink_input_index,
        level,
        max_depth=max_depth,
    ).name

    label = f"{source_output_index} -> {sink_input_index}"

    graphviz_graph.edge(
        source_name,
        sink_name,
        label=label,
        fontcolor="black" if level == 0 else "white",
    )


def _visualize(
    graphviz_graph,
    graph,
    *,
    max_depth=None,
    file_name=None,
    visualize_node,
    visualize_edge,
    level=0,
) -> graphviz.Digraph:

    if max_depth is not None:
        if max_depth < 1:
            raise RuntimeError("max_depth must be greater than 0")
        if level == max_depth:
            return graphviz_graph

    for node in graph:
        visualize_node(
            graphviz_graph,
            graph,
            node,
            max_depth,
            visualize_node,
            visualize_edge,
            level,
        )

    for node in graph:
        for edge in graph.in_edges(node, data=True, keys=True):
            visualize_edge(graphviz_graph, graph, edge, max_depth, level)

    if file_name is not None:
        file_name = pathlib.Path(file_name)
        if file_name.suffix != ".svg":
            raise ValueError(
                f"file_name must have a .svg suffix, not {file_name.suffix}"
            )
        format = file_name.suffix[1:]
        graphviz_graph.render(file_name.with_suffix(""), format=format)
        logger.info(f'Graph visualization saved to "{file_name}"')

    return graphviz_graph


def _flatten_graph(
    graph,
    *,
    new_graph=None,
    level=0,
) -> multidigraph.MultiDiGraph:

    if new_graph is None:
        new_graph = multidigraph.MultiDiGraph()

    for node in multidigraph.topological_traversal(graph):
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, TorchModule):
            module_graph = _flatten_graph(
                operation.graph, new_graph=new_graph, level=level + 1
            )
            new_graph = multidigraph.compose_all(new_graph, module_graph)
        else:
            new_graph = new_graph.add_node(
                node,
                **graph.nodes[node],
            )

    for node in multidigraph.topological_traversal(graph):
        operation = graph.nodes[node]["operation"]
        for source, sink, edge_data in graph.in_edges(node, data=True):
            source = get_source(graph, source, edge_data["source_output_index"], level)
            sink = get_sink(graph, sink, edge_data["sink_input_index"], level)
            if source not in new_graph or sink not in new_graph:
                continue
            new_graph = new_graph.add_edge(
                source,
                sink,
                **edge_data,
            )

    return new_graph


def flatten_graph(graph) -> multidigraph.MultiDiGraph:
    graph = _flatten_graph(graph)

    for node in multidigraph.topological_traversal(graph):
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, TorchModuleInput):
            ((predecessor, _, data),) = list(graph.in_edges(node, data=True))
            for _, sink, edge_data in graph.out_edges(node, data=True):
                edge_data = edge_data.set(
                    "source_output_index", data["source_output_index"]
                )
                graph = graph.add_edge(
                    predecessor,
                    sink,
                    **edge_data,
                )
            graph = graph.remove_node(node)
    return graph


def process_output(output):
    output_tensors = []
    if isinstance(output, TracedTensor):
        output_tensors.append(output)
    elif isinstance(output, (tuple, list)):
        for value in output:
            output_tensors += process_output(value)
    elif dataclasses.is_dataclass(output):
        for class_field in dataclasses.fields(output):
            value = getattr(output, class_field.name)
            output_tensors += process_output(value)
    elif isinstance(output, dict):
        for value in output.values():
            output_tensors += process_output(value)
    elif output is None:
        pass
    else:
        raise RuntimeError(f"Unexpected type {type(output)}")
    return output_tensors


def get_graph(
    value: Union[TracedTensor, Tuple[TracedTensor, ...]],
    as_networkx: bool = False,
    flatten: bool = False,
) -> multidigraph.MultiDiGraph:
    output_tensors = process_output(value)
    graph = multidigraph.compose_all(*[tensor.graph for tensor in output_tensors])
    if flatten:
        graph = flatten_graph(graph)
    if as_networkx:
        graph = multidigraph.to_networkx(graph)
    return graph


def visualize(
    output: Union[TracedTensor, Tuple[TracedTensor, ...]],
    *,
    show_modules: bool = True,
    max_depth: Optional[int] = None,
    file_name: Optional[Union[str, pathlib.Path]] = None,
    graph_attr: Optional[dict] = None,
    node_attr: Optional[dict] = None,
    edge_attr: Optional[dict] = None,
    visualize_node: Callable = visualize_node,
    visualize_edge: Callable = visualize_edge,
) -> graphviz.Digraph:

    if not show_modules and max_depth is not None:
        raise RuntimeError("max_depth is not supported with show_modules=True")

    graph = get_graph(output)
    if not show_modules:
        graph = flatten_graph(graph)

    if graph_attr is None:
        graph_attr = {"ordering": "in", "rankdir": "TB"}

    if node_attr is None:
        node_attr = {
            "style": "filled",
            "border": "1",
            "fontsize": "10",
            "ranksep": "0.1",
            "height": "0.2",
            "fontname": "Linux libertine",
            "margin": "0",
            "shape": "box",
        }
    if edge_attr is None:
        edge_attr = {
            "fontsize": "10",
        }

    graphviz_graph = graphviz.Digraph(
        engine="dot",
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )

    return _visualize(
        graphviz_graph,
        graph,
        max_depth=max_depth,
        file_name=file_name,
        visualize_node=visualize_node,
        visualize_edge=visualize_edge,
    )
