import torch

import torchtrail


def test_linear():

    module = torch.nn.Linear(128, 256)

    with torchtrail.trace():
        input_tensor = torch.rand(64, 128)
        output = module(input_tensor)

    torchtrail.visualize(output)
    assert len(torchtrail.get_graph(output)) == 2
    assert len(torchtrail.get_graph(output, flatten=True)) == 4

    codegen_output = torchtrail.codegen(output)
    assert len(codegen_output.split("\n")) == 12


def test_module_list():

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Conv2d(3, 3, 3) for _ in range(4)]
            )

        def forward(self, x):
            for module in self.layers:
                x = module(x)
            return x

    module = Module()

    with torchtrail.trace():
        output = module(torch.rand(1, 3, 100, 100))

    torchtrail.visualize(output)
    assert len(torchtrail.get_graph(output)) == 2
    assert len(torchtrail.get_graph(output, flatten=True)) == 13

    codegen_output = torchtrail.codegen(output)
    assert len(codegen_output.split("\n")) == 37


def test_module_list_with_multiple_traces():

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Conv2d(3, 3, 3) for _ in range(4)]
            )

        def forward(self, x):
            for module in self.layers:
                x = module(x)
            return x

    module = Module()

    with torchtrail.trace():
        input_tensor = torch.rand(1, 3, 100, 100)
        output = module(input_tensor)

    torchtrail.visualize(output)
    assert len(torchtrail.get_graph(output)) == 2
    assert len(torchtrail.get_graph(output, flatten=True)) == 13

    ignored_output = module(
        input_tensor
    )  # Output outside the context manager won't be traced

    with torchtrail.trace():
        # Tracing second time will merge the original graph with the new one
        another_output = module(input_tensor)

    torchtrail.visualize(another_output)
    assert len(torchtrail.get_graph(another_output)) == 3
    assert len(torchtrail.get_graph(another_output, flatten=True)) == 25

    codegen_output = torchtrail.codegen(output)
    assert len(codegen_output.split("\n")) == 37
