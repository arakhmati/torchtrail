import torch

import torchtrail


def test_rand_exp():
    with torchtrail.trace():
        tensor = torch.rand(1, 64)
        tensor = torch.exp(tensor)

    assert len(tensor.graph) == 2

    torchtrail.visualize(tensor)


def test_rand_add_exp():
    with torchtrail.trace():
        tensor = torch.rand(1, 64)
        tensor = tensor + tensor
        tensor = torch.exp(tensor)

    assert len(tensor.graph) == 3

    torchtrail.visualize(tensor)


def test_zeros():
    with torchtrail.trace():
        tensor = torch.zeros(1, 64)

    assert len(tensor.graph) == 1

    torchtrail.visualize(tensor)


def test_rand_split():
    with torchtrail.trace():
        tensor = torch.rand(1, 64)
        tensors = torch.split(tensor, split_size_or_sections=32, dim=1)

    for tensor in tensors:
        assert len(tensor.graph) == 2

    torchtrail.visualize(tensors)
