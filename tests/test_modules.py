import torch

import torchtrail


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
