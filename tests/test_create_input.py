import torch
import torchtrail


def test_create_input():
    tensor = torch.rand(1, 64)
    with torchtrail.trace():
        traced_tensor = torchtrail.create_input(tensor)
        output = torch.exp(traced_tensor)

    torchtrail.visualize(output)
    assert len(torchtrail.get_graph(output)) == 2
    codegen_output = torchtrail.codegen(output)
    assert len(codegen_output.split("\n")) == 6
