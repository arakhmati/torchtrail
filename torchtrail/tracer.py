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

from typing import Any, Callable, Optional, Union, Tuple

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
    topological_traversal,
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
        return "torch.Tensor"


class TorchParameter(PClass):
    parameter = field(mandatory=True)

    def __repr__(self):
        output = "torch.nn.Parameter"
        if hasattr(self.parameter, "torchtrail_name"):
            output = f"{output}\n{self.parameter.torchtrail_name}"
        return output


class TorchFunction(PClass):
    function = field(mandatory=True)

    def __repr__(self):
        return self.function.__name__


class TorchModule(PClass):
    module = field(mandatory=True)
    graph: MultiDiGraph = field(mandatory=True)
    inputs: list[TracedTorchTensor] = field(mandatory=True)
    outputs: list[TracedTorchTensor] = field(mandatory=True)

    def __repr__(self):
        output = type(self.module).__name__
        if hasattr(self.module, "torchtrail_name"):
            return f"{output}\n{self.module.torchtrail_name}"
        return output


class TorchModuleInput(PClass):
    def __repr__(self):
        return "torch.Tensor"


UNIQUE_ID = 0


def get_unique_id():
    global UNIQUE_ID
    output = UNIQUE_ID
    UNIQUE_ID += 1
    return output


def create_input(
    tensor: torch.Tensor, function: Optional[Callable[..., Any]] = None
) -> TracedTorchTensor:
    if isinstance(tensor, torch.nn.Parameter):
        node_name = f"torch_parameter_{get_unique_id()}"
        node = Node(name=node_name)
        graph = MultiDiGraph().add_node(
            node,
            operation=TorchParameter(parameter=tensor),
            shapes=(tuple(tensor.shape),),
            dtypes=(tensor.dtype,),
        )
        return TracedTorchTensor(tensor, graph=graph, node=node, output_index=0)
    elif isinstance(tensor, torch.Tensor):
        if function is None:
            operation = TorchTensor(tensor=tensor)
        else:
            operation = TorchFunction(function=function)
        node_name = f"torch_input_{get_unique_id()}"
        node = Node(name=node_name)
        graph = MultiDiGraph().add_node(
            node,
            operation=operation,
            shapes=(tuple(tensor.shape),),
            dtypes=(tensor.dtype,),
        )
        return TracedTorchTensor(tensor, graph=graph, node=node, output_index=0)
    else:
        raise ValueError(f"Unknown input type: {type(tensor)}")


def preprocess_args_and_kwargs(*args, **kwargs) -> Any:
    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, TracedTorchTensor):
            return arg
        elif isinstance(arg, torch.Tensor):
            return create_input(arg)
        else:
            return arg

    args = [preprocess_arg(arg) for arg in args]
    kwargs = {name: preprocess_arg(arg) for name, arg in kwargs.items()}
    return args, kwargs


class LazyTensor:
    def __init__(self, graph: MultiDiGraph, node: Node, output_index: int):
        self.graph: MultiDiGraph = graph
        self.node: Node = node
        self.output_index: int = output_index

    @classmethod
    def from_traced_tesnor(cls, traced_tesnor: TracedTorchTensor):
        return cls(traced_tesnor.graph, traced_tesnor.node, traced_tesnor.output_index)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def shape(self) -> str:
        return self.graph.nodes[self.node]["shapes"][self.output_index]


class TracedTorchTensor(torch.Tensor):
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

        if kwargs is None:
            kwargs = {}

        args, kwargs = preprocess_args_and_kwargs(*args, **kwargs)

        def get_input_tensors(object):
            input_tensors = []
            if isinstance(object, TracedTorchTensor):
                input_tensors.append(object)
            elif isinstance(object, (list, tuple)):
                for element in object:
                    input_tensors += get_input_tensors(element)
            elif isinstance(object, dict):
                for value in object.values():
                    input_tensors += get_input_tensors(value)
            return input_tensors

        input_tensors = get_input_tensors(args) + get_input_tensors(kwargs)

        # This is necessary for torch version < 1.10
        output = super().__torch_function__(function, types, args, kwargs)

        if output is None:
            return
        if isinstance(output, (int, torch.Size, torch.device, torch.dtype, str)):
            return output
        elif isinstance(output, torch.Tensor) and not isinstance(
            output, TracedTorchTensor
        ):
            raise ValueError(f"Expected torch.Tensor but got {type(output)}")
        else:
            if not isinstance(output, TracedTorchTensor):
                raise ValueError(f"Expected TracedTorchTensor but got {type(output)}")

        node_name = f"{function.__name__}_{get_unique_id()}"
        node = Node(name=node_name)
        graph = merge_graphs(*((tensor.graph, tensor.node) for tensor in input_tensors))
        graph = graph.add_node(
            node,
            operation=TorchFunction(function=function),
            shapes=(tuple(output.shape),),
            dtypes=(output.dtype,),
        )
        for input_index, tensor in enumerate(input_tensors):
            graph = graph.add_edge(
                tensor.node,
                node,
                source_output_index=tensor.output_index,
                sink_input_index=input_index,
            )
        return TracedTorchTensor(output, graph=graph, node=node, output_index=0)


def wrap_create_function(function: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> TracedTorchTensor:
        input_tensor = function(*args, **kwargs)
        return create_input(input_tensor, function)

    return wrapper


def create_module_input(tensor: torch.Tensor) -> TracedTorchTensor:
    node_name = f"module_input_{get_unique_id()}"
    node = Node(name=node_name)
    graph = MultiDiGraph().add_node(
        node,
        operation=TorchModuleInput(),
        shapes=(tuple(tensor.shape),),
        dtypes=(tensor.dtype,),
    )
    return TracedTorchTensor(tensor, graph=graph, node=node, output_index=0)


def convert_to_module_args_and_kwargs(*args, **kwargs) -> Any:
    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, TracedTorchTensor):
            return create_module_input(arg)
        elif isinstance(arg, torch.nn.Parameter):
            return create_module_input(arg)
        else:
            return arg

    args = [preprocess_arg(arg) for arg in args]
    kwargs = {name: preprocess_arg(arg) for name, arg in kwargs.items()}
    return args, kwargs


def traced_module_forward(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:

    if not hasattr(module, "torchtrail_name"):
        module.torchtrail_name = ""

    for name, child in module.named_modules():
        if not hasattr(child, "torchtrail_name"):
            child.torchtrail_name = name

    for name, parameter in module.named_parameters():
        if not hasattr(parameter, "torchtrail_name"):
            parameter.torchtrail_name = name

    args, kwargs = preprocess_args_and_kwargs(*args, **kwargs)

    module_args, module_kwargs = convert_to_module_args_and_kwargs(*args, **kwargs)
    output = TORCH_NN_MODULE_CALL(module, *module_args, **module_kwargs)

    module_outputs = []
    if isinstance(output, torch.Tensor) and not isinstance(output, TracedTorchTensor):
        raise ValueError(f"Expected torch.Tensor but got {type(output)}")
    elif isinstance(output, tuple):
        # TODO: support nested tuples
        tensors = []
        for element in output:
            if element is None:
                continue
            if isinstance(element, TracedTorchTensor):
                tensors.append(element)
                continue
            raise ValueError(f"Unexpected element in a tuple: {type(element)}")
        module_outputs = tensors
    elif dataclasses.is_dataclass(output):
        # TODO: support nested dataclasses
        output_index = 0
        for class_field in dataclasses.fields(output):
            value = getattr(output, class_field.name)
            if isinstance(value, TracedTorchTensor):
                module_outputs.append(value)
    else:
        if not isinstance(output, TracedTorchTensor):
            raise ValueError(f"Expected TracedTorchTensor but got {type(output)}")
        module_outputs.append(output)

    module_inputs = [
        LazyTensor.from_traced_tesnor(arg)
        for arg in module_args
        if isinstance(arg, TracedTorchTensor)
    ]
    module_inputs += [
        LazyTensor.from_traced_tesnor(arg)
        for arg in module_kwargs.values()
        if isinstance(arg, TracedTorchTensor)
    ]

    module_outputs = [
        LazyTensor.from_traced_tesnor(tensor) for tensor in module_outputs
    ]
    module_graph = compose_all(
        *[tensor.graph for tensor in module_inputs + module_outputs]
    )

    operation = TorchModule(
        module=module, graph=module_graph, inputs=module_inputs, outputs=module_outputs
    )

    input_tensors = [arg for arg in args if isinstance(arg, TracedTorchTensor)]
    input_tensors += [
        arg for arg in kwargs.values() if isinstance(arg, TracedTorchTensor)
    ]

    node_name = f"{module.torchtrail_name}_{get_unique_id()}"
    node = Node(name=node_name)

    graph = merge_graphs(*((tensor.graph, tensor.node) for tensor in input_tensors))

    def create_output_tensor(
        output_tensor: torch.Tensor, output_index
    ) -> TracedTorchTensor:
        nonlocal graph
        if node in graph:
            shapes = graph.nodes[node]["shapes"] + (tuple(output_tensor.shape),)
            dtypes = graph.nodes[node]["dtypes"] + (output_tensor.dtype,)
        else:
            shapes = (tuple(output_tensor.shape),)
            dtypes = (output_tensor.dtype,)
        graph = graph.add_node(
            node,
            operation=operation,
            shapes=shapes,
            dtypes=dtypes,
        )
        for input_index, tensor in enumerate(input_tensors):
            graph = graph.add_edge(
                tensor.node,
                node,
                source_output_index=tensor.output_index,
                sink_input_index=input_index,
            )
        output_tensor = TracedTorchTensor(
            output_tensor, graph=graph, node=node, output_index=output_index
        )
        return output_tensor

    if isinstance(output, torch.Tensor) and not isinstance(output, TracedTorchTensor):
        raise ValueError(f"Expected torch.Tensor but got {type(output)}")
    elif isinstance(output, tuple):
        # TODO: support nested tuples
        tensors = []
        for element in output:
            if element is None:
                continue
            if isinstance(element, TracedTorchTensor):
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
        for class_field in dataclasses.fields(output):
            value = getattr(output, class_field.name)
            if isinstance(value, TracedTorchTensor):
                value = create_output_tensor(value, output_index=output_index)
                output_index += 1
            updated_fields[class_field.name] = value
        return type(output)(**updated_fields)
    else:
        if not isinstance(output, TracedTorchTensor):
            raise ValueError(f"Expected TracedTorchTensor but got {type(output)}")
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


def get_source_name(graph, node, source_output_index, level, level_prefix, max_depth):
    name = f"{level_prefix}{node.name}"
    operation = graph.nodes[node]["operation"]

    if max_depth is not None and level + 1 == max_depth:
        return name

    if not isinstance(operation, TorchModule):
        return name

    module = operation
    module_graph = module.graph
    module_node = module.outputs[source_output_index].node
    return get_source_name(
        module_graph,
        module_node,
        source_output_index,
        level + 1,
        level_prefix=f"{name}/",
        max_depth=max_depth,
    )


def get_sink_name(graph, node, sink_input_index, level, level_prefix, max_depth):
    name = f"{level_prefix}{node.name}"
    operation = graph.nodes[node]["operation"]

    if max_depth is not None and level + 1 == max_depth:
        return name

    if not isinstance(operation, TorchModule):
        return name

    module = operation
    module_graph = module.graph
    module_node = module.inputs[sink_input_index].node
    return get_sink_name(
        module_graph,
        module_node,
        sink_input_index,
        level + 1,
        level_prefix=f"{name}/",
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
        name = f"{level_prefix}{node.name}"
        if isinstance(operation, TorchModule):

            color = LEVEL_COLORS[level % len(LEVEL_COLORS)]
            if max_depth is None or level < max_depth - 1:
                with graphviz_graph.subgraph(name=node.name) as cluster_graph:
                    cluster_graph.attr(
                        fontcolor="black",
                        bgcolor=color,
                        cluster="true",
                        label=f"{operation}",
                        rankdir="TB",
                        shape="hexagon",
                    )
                    cluster_graph.node_attr["style"] = "filled"
                    _visualize(
                        operation.graph,
                        max_depth=max_depth,
                        graphviz_graph=cluster_graph,
                        level=level + 1,
                        level_prefix=f"{name}/",
                    )
            else:
                shapes = attributes["shapes"]
                dtypes = attributes["dtypes"]
                shapes = shapes[0] if len(shapes) == 1 else shapes
                dtypes = dtypes[0] if len(dtypes) == 1 else dtypes
                graphviz_graph.node(
                    name,
                    label=f"{operation}\n{shapes}\n{dtypes}",
                    fontcolor="black",
                    fillcolor=color,
                    shape="hexagon",
                )

        else:
            if isinstance(operation, (TorchTensor, TorchParameter, TorchModuleInput)):
                shape = "box"
            else:
                shape = "circle"

            shapes = attributes["shapes"]
            dtypes = attributes["dtypes"]
            shapes = shapes[0] if len(shapes) == 1 else shapes
            dtypes = dtypes[0] if len(dtypes) == 1 else dtypes
            graphviz_graph.node(
                name,
                label=f"{operation}\n{shapes}\n{dtypes}",
                fillcolor="white",
                shape=shape,
            )

    def visualize_edge(graphviz_graph, graph, edge):
        source, sink, _, edge_data = edge

        source_output_index = edge_data["source_output_index"]
        sink_input_index = edge_data["sink_input_index"]

        source_name = get_source_name(
            graph,
            source,
            source_output_index,
            level,
            level_prefix=level_prefix,
            max_depth=max_depth,
        )
        sink_name = get_sink_name(
            graph,
            sink,
            sink_input_index,
            level,
            level_prefix=level_prefix,
            max_depth=max_depth,
        )

        label = f"{source_output_index} -> {sink_input_index}"

        graphviz_graph.edge(
            source_name,
            sink_name,
            label=label,
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
    value: Union[TracedTorchTensor, Tuple[TracedTorchTensor, ...]],
    *,
    max_depth: Optional[int] = None,
    file_name: Optional[str] = None,
) -> graphviz.Digraph:
    if isinstance(value, TracedTorchTensor):
        graph = value.graph
    elif isinstance(value, tuple):
        graph = compose_all(*[tensor.graph for tensor in value])
    else:
        raise ValueError(f"Unexpected input type: {type(value)}")
    return _visualize(graph, max_depth=max_depth, file_name=file_name)
