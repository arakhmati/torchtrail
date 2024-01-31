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

from typing import Any, Callable, Optional

import dataclasses
from contextlib import contextmanager

import graphviz
import torch
from pyrsistent import PClass, field

from torchtrail.multidigraph import (
    MultiDiGraph,
    compose_all,
    merge_graphs,
    visualize_graph,
)


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
    tensor = field(mandatory=True)

    def __repr__(self):
        return "TorchTensor"


class TorchParameter(PClass):
    parameter = field(mandatory=True)

    def __repr__(self):
        return f"TorchParameter"


class TorchFunction(PClass):
    function = field(mandatory=True)

    def __repr__(self):
        return self.function.__name__


class TorchModule(PClass):
    module = field(mandatory=True)
    graph: MultiDiGraph = field(mandatory=True)
    inputs: list[TorchTrailTensor] = field(mandatory=True)
    outputs: list[TorchTrailTensor] = field(mandatory=True)

    def __repr__(self):
        return type(self.module).__name__


class TorchModuleInput(PClass):
    def __repr__(self):
        return "TorchModuleInput"


UNIQUE_ID = 0


def get_unique_id():
    global UNIQUE_ID
    output = UNIQUE_ID
    UNIQUE_ID += 1
    return output


def create_input(
    tensor: torch.Tensor, function: Optional[Callable[..., Any]] = None
) -> TorchTrailTensor:
    if isinstance(tensor, torch.nn.Parameter):
        node_name = f"torch_parameter_{get_unique_id()}"
        node = Node(name=node_name)
        graph = MultiDiGraph().add_node(
            node,
            operation=TorchParameter(parameter=tensor),
            shapes=(tuple(tensor.shape),),
        )
        return TorchTrailTensor(tensor, graph=graph, node=node, output_index=0)
    elif isinstance(tensor, torch.Tensor):
        if function is None:
            operation = TorchTensor(tensor=tensor)
        else:
            operation = TorchFunction(function=function)
        node_name = f"torch_input_{get_unique_id()}"
        node = Node(name=node_name)
        graph = MultiDiGraph().add_node(
            node, operation=operation, shapes=(tuple(tensor.shape),)
        )
        return TorchTrailTensor(tensor, graph=graph, node=node, output_index=0)
    else:
        raise ValueError(f"Unknown input type: {type(tensor)}")


def preprocess_args_and_kwargs(*args, **kwargs) -> Any:
    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, TorchTrailTensor):
            return arg
        elif isinstance(arg, torch.Tensor):
            return create_input(arg)
        else:
            return arg

    args = [preprocess_arg(arg) for arg in args]
    kwargs = {name: preprocess_arg(arg) for name, arg in kwargs.items()}
    return args, kwargs


class TorchTrailTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls: Any,
        tensor: Any,
        graph: MultiDiGraph,
        node: Node,
        output_index: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return super().__new__(cls, tensor, *args, **kwargs)  # type: ignore[call-arg]

    def __init__(
        self, tensor: Any, *, graph: MultiDiGraph, node: Node, output_index: int
    ):
        self.graph: MultiDiGraph = graph
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
        # print(function.__name__)

        if kwargs is None:
            kwargs = {}

        args, kwargs = preprocess_args_and_kwargs(*args, **kwargs)

        def get_input_tensors(object):
            input_tensors = []
            if isinstance(object, TorchTrailTensor):
                input_tensors.append(object)
            elif isinstance(object, (list, tuple)):
                for element in object:
                    input_tensors += get_input_tensors(element)
            elif isinstance(object, dict):
                for value in object.values():
                    input_tensors += get_input_tensors(value)
            return input_tensors

        input_tensors = get_input_tensors(args) + get_input_tensors(kwargs)

        # print(f"\tinput_tensors: {len(input_tensors)}")

        # This is necessary for torch version < 1.10
        output = super().__torch_function__(function, types, args, kwargs)
        # print("\toutput type: ", type(output))

        if output is None:
            return
        if isinstance(output, (int, torch.Size, torch.device, torch.dtype, str)):
            return output
        elif isinstance(output, torch.Tensor) and not isinstance(
            output, TorchTrailTensor
        ):
            raise ValueError(f"Expected torch.Tensor but got {type(output)}")
        else:
            if not isinstance(output, TorchTrailTensor):
                raise ValueError(f"Expected TorchTrailTensor but got {type(output)}")

        node_name = f"{function.__name__}_{get_unique_id()}"
        node = Node(name=node_name)
        graph = merge_graphs(*((tensor.graph, tensor.node) for tensor in input_tensors))
        graph = graph.add_node(
            node,
            operation=TorchFunction(function=function),
            shapes=(tuple(output.shape),),
        )
        for input_index, tensor in enumerate(input_tensors):
            graph = graph.add_edge(
                tensor.node,
                node,
                source_output_index=tensor.output_index,
                sink_input_index=input_index,
            )
        return TorchTrailTensor(output, graph=graph, node=node, output_index=0)


class LazyTensor:
    def __init__(self, graph: MultiDiGraph, node: Node, output_index: int):
        self.graph: MultiDiGraph = graph
        self.node: Node = node
        self.output_index: int = output_index

    @classmethod
    def from_tracer_tensor(cls, tracer_tensor: TorchTrailTensor):
        return cls(tracer_tensor.graph, tracer_tensor.node, tracer_tensor.output_index)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def shape(self) -> str:
        return self.graph.nodes[self.node]["shapes"][self.output_index]


def wrap_create_function(function: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> TorchTrailTensor:
        # print(f"creation_function: {function.__name__}")
        input_tensor = function(*args, **kwargs)
        return create_input(input_tensor, function)

    return wrapper


def create_module_input(input_tensor: torch.Tensor) -> TorchTrailTensor:
    node_name = f"module_input_{get_unique_id()}"
    node = Node(name=node_name)
    graph = MultiDiGraph().add_node(
        node, operation=TorchModuleInput(), shapes=(tuple(input_tensor.shape),)
    )
    return TorchTrailTensor(input_tensor, graph=graph, node=node, output_index=0)


def convert_to_module_args_and_kwargs(*args, **kwargs) -> Any:
    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, TorchTrailTensor):
            return create_module_input(arg)
        elif isinstance(arg, torch.nn.Parameter):
            return create_module_input(arg)
        else:
            return arg

    args = [preprocess_arg(arg) for arg in args]
    kwargs = {name: preprocess_arg(arg) for name, arg in kwargs.items()}
    return args, kwargs


def traced_module_forward(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
    # print(module.__class__.__name__)

    args, kwargs = preprocess_args_and_kwargs(*args, **kwargs)

    module_args, module_kwargs = convert_to_module_args_and_kwargs(*args, **kwargs)
    output = TORCH_NN_MODULE_CALL(module, *module_args, **module_kwargs)
    # print("\toutput type", type(output))

    module_outputs = []
    if isinstance(output, torch.Tensor) and not isinstance(output, TorchTrailTensor):
        raise ValueError(f"Expected torch.Tensor but got {type(output)}")
    elif isinstance(output, tuple):
        # TODO: support nested tuples
        tensors = []
        for element in output:
            if element is None:
                continue
            if isinstance(element, TorchTrailTensor):
                tensors.append(element)
                continue
            raise ValueError(f"Unexpected element in a tuple: {type(element)}")
        module_outputs = tensors
    elif dataclasses.is_dataclass(output):
        # TODO: support nested dataclasses
        output_index = 0
        for field in dataclasses.fields(output):
            value = getattr(output, field.name)
            if isinstance(value, TorchTrailTensor):
                module_outputs.append(value)
    else:
        if not isinstance(output, TorchTrailTensor):
            raise ValueError(f"Expected TorchTrailTensor but got {type(output)}")
        module_outputs.append(output)

    module_inputs = [
        LazyTensor.from_tracer_tensor(arg)
        for arg in module_args
        if isinstance(arg, TorchTrailTensor)
    ]
    module_inputs += [
        LazyTensor.from_tracer_tensor(arg)
        for arg in module_kwargs.values()
        if isinstance(arg, TorchTrailTensor)
    ]
    module_outputs = [
        LazyTensor.from_tracer_tensor(tensor) for tensor in module_outputs
    ]
    module_graph = compose_all(*[tensor.graph for tensor in module_outputs])

    operation = TorchModule(
        module=module, graph=module_graph, inputs=module_inputs, outputs=module_outputs
    )

    input_tensors = [arg for arg in args if isinstance(arg, TorchTrailTensor)]
    input_tensors += [
        arg for arg in kwargs.values() if isinstance(arg, TorchTrailTensor)
    ]

    node_name = f"{module.__class__.__name__}_{get_unique_id()}"
    node = Node(name=node_name)

    def create_output_tensor(
        output_tensor: torch.Tensor, output_index
    ) -> TorchTrailTensor:
        graph = merge_graphs(*((tensor.graph, tensor.node) for tensor in input_tensors))
        graph = graph.add_node(
            node, operation=operation, shapes=(tuple(output_tensor.shape),)
        )
        for input_index, tensor in enumerate(input_tensors):
            graph = graph.add_edge(
                tensor.node,
                node,
                source_output_index=tensor.output_index,
                sink_input_index=input_index,
            )
        output_tensor = TorchTrailTensor(
            output_tensor, graph=graph, node=node, output_index=output_index
        )
        return output_tensor

    if isinstance(output, torch.Tensor) and not isinstance(output, TorchTrailTensor):
        raise ValueError(f"Expected torch.Tensor but got {type(output)}")
    elif isinstance(output, tuple):
        # TODO: support nested tuples
        tensors = []
        for element in output:
            if element is None:
                continue
            if isinstance(element, TorchTrailTensor):
                tensors.append(element)
                continue
            raise ValueError(f"Unexpected element in a tuple: {type(element)}")
        return tuple(
            create_output_tensor(tensor, output_index)
            for output_index, tensor in enumerate(tensors)
        )
    elif dataclasses.is_dataclass(output):
        # TODO: support nested dataclasses
        output_index = 0
        updated_fields = {}
        for field in dataclasses.fields(output):
            value = getattr(output, field.name)
            if isinstance(value, TorchTrailTensor):
                value = create_output_tensor(value, output_index=output_index)
                output_index += 1
            updated_fields[field.name] = value
        return type(output)(**updated_fields)
    else:
        if not isinstance(output, TorchTrailTensor):
            raise ValueError(f"Expected TorchTrailTensor but got {type(output)}")
        return create_output_tensor(output, output_index=0)


@contextmanager
def trace(*, trace_modules=True):

    # Monkey-patch module __call__ and torch creation ops
    if trace_modules:
        setattr(torch.nn.Module, "__call__", traced_module_forward)

    for name, op in zip(TORCH_CREATION_OPERATION_NAMES, TORCH_CREATION_OPERATIONS):
        setattr(torch, name, wrap_create_function(op))

    yield

    # Reset monkey-patched module __call__ and torch creation ops
    if trace_modules:
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


def get_source_name(graph, node, edge_data, local_level, prefix, max_depth):
    name = f"{prefix}_{node.name}_{local_level}"
    operation = graph.nodes[node]["operation"]

    if max_depth is not None and local_level + 1 == max_depth:
        return name

    if not isinstance(operation, TorchModule):
        return name

    module = operation
    module_graph = module.graph
    module_node = module.outputs[edge_data["source_output_index"]].node
    return get_source_name(
        module_graph,
        module_node,
        edge_data,
        local_level + 1,
        prefix=name,
        max_depth=max_depth,
    )


def get_sink_name(graph, node, edge_data, local_level, prefix, max_depth):
    name = f"{prefix}_{node.name}_{local_level}"
    operation = graph.nodes[node]["operation"]

    if max_depth is not None and local_level + 1 == max_depth:
        return name

    if not isinstance(operation, TorchModule):
        return name

    module = operation
    module_graph = module.graph
    module_node = module.inputs[edge_data["sink_input_index"]].node
    return get_sink_name(
        module_graph,
        module_node,
        edge_data,
        local_level + 1,
        prefix=name,
        max_depth=max_depth,
    )


def _visualize(
    graph,
    *,
    max_depth=None,
    file_name=None,
    graphviz_graph=None,
    level=0,
    level_prefix="",
) -> graphviz.Digraph:

    if max_depth is not None:
        if max_depth < 1:
            raise ValueError("max_depth must be greater than 0")
        if level == max_depth:
            return graphviz_graph

    if graphviz_graph is None:
        graph_attr = {"ordering": "in", "rankdir": "TB"}
        node_attr = {
            "style": "filled",
            "shape": "plaintext",
            "align": "left",
            "fontsize": "10",
            "ranksep": "0.1",
            "height": "0.2",
            "fontname": "Linux libertine",
            "margin": "0",
        }
        edge_attr = {
            "fontsize": "10",
        }
        graphviz_graph = graphviz.Digraph(
            engine="dot",
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )

    def visualize_node(graphviz_graph, graph, node):
        attributes = graph.nodes[node]
        operation = attributes["operation"]
        name = f"{level_prefix}_{node.name}_{level}"
        if isinstance(operation, TorchModule):
            if max_depth is None or level < max_depth - 1:
                with graphviz_graph.subgraph(name=node.name) as cluster_graph:
                    cluster_graph.attr(
                        fontcolor="black",
                        bgcolor=LEVEL_COLORS[level % len(LEVEL_COLORS)],
                        cluster="true",
                        label=f"{operation}",
                        rankdir="TB",
                    )
                    cluster_graph.node_attr["style"] = "filled"
                    cluster_graph.node_attr["fillcolor"] = "white"
                    _visualize(
                        operation.graph,
                        max_depth=max_depth,
                        graphviz_graph=cluster_graph,
                        level=level + 1,
                        level_prefix=name,
                    )
            else:
                shapes = attributes["shapes"]
                shapes = shapes[0] if len(shapes) == 1 else shapes
                graphviz_graph.node(name, label=f"{operation}\n{shapes}")

        else:
            shapes = attributes["shapes"]
            shapes = shapes[0] if len(shapes) == 1 else shapes
            graphviz_graph.node(name, label=f"{operation}\n{shapes}")

    def visualize_edge(graphviz_graph, graph, edge):
        source, sink, _, edge_data = edge

        source_name = get_source_name(
            graph, source, edge_data, level, prefix=level_prefix, max_depth=max_depth
        )
        sink_name = get_sink_name(
            graph, sink, edge_data, level, prefix=level_prefix, max_depth=max_depth
        )

        graphviz_graph.edge(
            source_name,
            sink_name,
            label=f"{edge_data['source_output_index']} -> {edge_data['sink_input_index']}",
            fontcolor="black" if level == 0 else "white",
        )

    return visualize_graph(
        graph,
        graphviz_graph=graphviz_graph,
        visualize_node=visualize_node,
        visualize_edge=visualize_edge,
        file_name=file_name if level == 0 else None,
    )


def visualize(
    value: Union[TorchTrailTensor, Tuple[TorchTrailTensor, ...]],
    *,
    max_depth: Optional[int] = None,
    file_name: Optional[str] = None,
) -> graphviz.Digraph:
    if isinstance(value, TorchTrailTensor):
        graph = value.graph
    elif isinstance(value, tuple):
        graph = merge_graphs(*[tensor.graph for tensor in value])
    else:
        raise ValueError(f"Unexpected input type: {type(value)}")
    return _visualize(graph, max_depth=max_depth, file_name=file_name)
