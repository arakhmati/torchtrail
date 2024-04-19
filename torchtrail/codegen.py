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

import io

import networkx as nx

import torchtrail


def get_module_input_nodes(module_operation):
    return [module_input.node for module_input in module_operation.inputs]


def get_module_outputs(module_operation):
    return [
        (module_output.node, module_output.output_index)
        for module_output in module_operation.outputs
    ]


def get_module_input_tensor_names(module_operation):
    module_input_nodes = get_module_input_nodes(module_operation)
    input_tensor_names = []
    for node in module_input_nodes:
        operation = module_operation.graph.nodes[node]["operation"]
        input_tensor_names.append(f"{operation.name}")
    return input_tensor_names


def get_module_name(operation):
    module = operation.module

    output = f"{type(module).__name__}"
    if hasattr(module, "torchtrail_name"):
        if module.torchtrail_name is not "":
            torchtrail_name = module.torchtrail_name.replace(".", "_")
            return f"{output}_{torchtrail_name}"
    return output


def node_to_statement(
    string_io, graph, node, input_variables, output_variables, prefix
):
    def process_arguments(argument_name_values):
        def process_value(value):
            if isinstance(value, torchtrail.tracer.InputTensorIndex):
                return input_variables[value.index]
            elif isinstance(value, (tuple, list)):
                joined_elements = ", ".join([process_value(v) for v in value])
                return f"[{joined_elements}]"
            elif isinstance(value, dict):
                joined_elements = ", ".join(
                    [f"{key}: {process_value(value)}" for key, value in value.items()]
                )
                return f"{{{joined_elements}}}"
            else:
                return f"{value}"

        return [(name, process_value(value)) for name, value in argument_name_values]

    operation = graph.nodes[node]["operation"]
    shapes = graph.nodes[node]["shapes"]
    dtypes = graph.nodes[node]["dtypes"]
    duration = graph.nodes[node].get("duration", None)

    if not output_variables:
        assignment_statement = ""
    elif len(output_variables) == 1:
        assignment_statement = f"{output_variables[0]} = "
    else:
        assignment_statement = f"{', '.join(output_variables)} = "

    if isinstance(operation, torchtrail.tracer.TorchParameter):
        torchtrail_name = operation.parameter.torchtrail_name
        torchtrail_name = torchtrail_name.replace(f"{prefix}.", "", 1)
        string_io.write(f"    {assignment_statement}parameters.{torchtrail_name}")

    elif isinstance(operation, torchtrail.tracer.TorchTensor):
        string_io.write(
            f"    {assignment_statement}torch.as_tensor({operation.tensor.flatten().tolist()[:8]}, ...).reshape({tuple(operation.tensor.shape)}).to({operation.tensor.dtype})"
        )

    elif isinstance(operation, torchtrail.tracer.TorchFunction):
        function_args = []
        function_kwargs = []
        for argument_name, argument in process_arguments(operation.arguments):
            if isinstance(argument_name, int):
                function_args.append(f"{argument}")
            else:
                function_kwargs.append(f"{argument_name}={argument}")

        arguments_string = []
        if function_args:
            arguments_string.append(", ".join(function_args))
        if function_kwargs:
            arguments_string.append(", ".join(function_kwargs))
        arguments_string = ", ".join(arguments_string)

        string_io.write(f"    {assignment_statement}{operation}({arguments_string})")

    elif isinstance(operation, torchtrail.tracer.TorchModule):
        module_name = get_module_name(operation)
        torchtrail_name = operation.module.torchtrail_name
        torchtrail_name = torchtrail_name.replace(f"{prefix}.", "", 1)

        function_args = []
        function_kwargs = []
        for argument_name, argument in process_arguments(operation.arguments):
            if isinstance(argument_name, int):
                function_args.append(f"{argument}")
            else:
                function_kwargs.append(f"{argument_name}={argument}")

        arguments_string = []
        if function_args:
            arguments_string.append(", ".join(function_args))
        if function_kwargs:
            arguments_string.append(", ".join(function_kwargs))
        arguments_string = ", ".join(arguments_string)

        if torchtrail_name == "":
            string_io.write(
                f"    {assignment_statement}{module_name}(config, {arguments_string}, parameters=parameters)"
            )
        else:
            string_io.write(
                f"    {assignment_statement}{module_name}(config, {arguments_string}, parameters=parameters.{torchtrail_name})"
            )
    else:
        raise ValueError(f"Unknown operation type: {operation}")

    if len(shapes) == 1:
        shapes = shapes[0]
    if len(dtypes) == 1:
        dtypes = dtypes[0]

    string_io.write(f"    # shapes: {shapes}, dtypes: {dtypes}")
    if duration is not None:
        string_io.write(f"; duration: {torchtrail.tracer.duration_to_string(duration)}")
    string_io.write("\n")


def return_statement(string_io, outputs, node_output_to_variable):
    string_io.write("    return ")
    for index, (output_node, output_index) in enumerate(outputs):
        node_output_variables = []
        node_output_variables.append(
            node_output_to_variable[(output_node, output_index)]
        )
        string_io.write(f"{node_output_variables[0]}")
        if index != len(outputs) - 1:
            string_io.write(", ")
    string_io.write("\n\n")


def module_to_source_code(string_io, module_operation, prefix=""):
    graph = module_operation.graph

    input_tensor_names = get_module_input_tensor_names(module_operation)
    input_tensor_names_as_string = ", ".join(input_tensor_names)

    module_name = get_module_name(module_operation)

    string_io.write(
        f"def {module_name}(config, {input_tensor_names_as_string}, *, parameters):\n"
    )

    module_input_nodes = get_module_input_nodes(module_operation)
    node_output_to_variable = {}
    for module_input, name in zip(module_input_nodes, input_tensor_names):
        node_output_to_variable[(module_input, 0)] = name

    index = 0
    for node in nx.topological_sort(graph):
        if node in module_input_nodes:
            continue

        input_nodes = [
            (input_node, edge_data)
            for input_node, _, edge_data in graph.in_edges(node, data=True)
        ]
        input_nodes = sorted(input_nodes, key=lambda x: x[1]["sink_input_index"])
        input_variables = [
            node_output_to_variable[(input_node, edge_data["source_output_index"])]
            for input_node, edge_data in input_nodes
        ]

        output_variables = []
        for output_index, _ in enumerate(graph.nodes[node]["shapes"]):
            variable = f"variable_{index}"
            index += 1
            output_variables.append(variable)
            node_output_to_variable[(node, output_index)] = variable

        node_to_statement(
            string_io, graph, node, input_variables, output_variables, prefix
        )

    return_statement(
        string_io, get_module_outputs(module_operation), node_output_to_variable
    )


def codegen_submodules(string_io, graph):
    for node in nx.topological_sort(graph):
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, torchtrail.tracer.TorchModule):
            module = operation.module
            codegen_submodules(string_io, operation.graph)
            module_to_source_code(string_io, operation, module.torchtrail_name)


def codegen_top_level(string_io, graph, output, top_level_name):
    string_io.write(f"def {top_level_name}():\n")

    node_output_to_variable = {}
    index = 0
    for node in nx.topological_sort(graph):
        input_nodes = [
            (input_node, edge_data)
            for input_node, _, edge_data in graph.in_edges(node, data=True)
        ]
        input_nodes = sorted(input_nodes, key=lambda x: x[1]["sink_input_index"])
        input_variables = [
            node_output_to_variable[(input_node, edge_data["source_output_index"])]
            for input_node, edge_data in input_nodes
        ]

        output_variables = []
        for output_index, _ in enumerate(graph.nodes[node]["shapes"]):
            variable = f"variable_{index}"
            index += 1
            output_variables.append(variable)
            node_output_to_variable[(node, output_index)] = variable
        node_to_statement(
            string_io, graph, node, input_variables, output_variables, prefix=""
        )

    output_tensors = torchtrail.tracer.process_output(output)
    output_node_and_index = [
        (tensor.node, tensor.output_index) for tensor in output_tensors
    ]
    return_statement(string_io, output_node_and_index, node_output_to_variable)


def codegen(output, top_level_name="main"):
    graph = torchtrail.tracer.get_graph(output)
    string_io = io.StringIO()
    codegen_submodules(string_io, graph)
    codegen_top_level(string_io, graph, output, top_level_name)
    return_value = string_io.getvalue()
    return return_value
