import torch

import torchtrail


def test_rand_exp():
    with torchtrail.trace():
        input_tensor = torch.rand(1, 64)
        output_tensor = torch.exp(input_tensor)

    torchtrail.visualize(output_tensor)
    assert len(torchtrail.get_graph(output_tensor)) == 2


def test_rand_add_exp():
    with torchtrail.trace():
        input_tensor = torch.rand(1, 64)
        output_tensor = input_tensor + input_tensor
        output_tensor = torch.exp(output_tensor)

    torchtrail.visualize(output_tensor)
    assert len(torchtrail.get_graph(output_tensor)) == 3


def test_zeros():
    with torchtrail.trace():
        input_tensor = torch.zeros(1, 64)

    torchtrail.visualize(input_tensor)
    assert len(torchtrail.get_graph(input_tensor)) == 1


def test_rand_split():
    with torchtrail.trace():
        input_tensor = torch.rand(1, 64)
        output_tensors = torch.split(input_tensor, split_size_or_sections=32, dim=1)

    torchtrail.visualize(output_tensors)
    assert len(torchtrail.get_graph(output_tensors)) == 2


def test_rand_exp_backward():
    with torchtrail.trace():
        input_tensor = torch.rand(1, 64)
        input_tensor.requires_grad = True
        output_tensor = torch.exp(input_tensor)
        output_tensor.backward(torch.ones_like(output_tensor))

    torchtrail.visualize(input_tensor.grad)
    assert len(torchtrail.get_graph(input_tensor.grad)) == 2
