import pytest

import torch
from torchvision import models

import torchtrail


@pytest.mark.parametrize("show_modules", [True, False])
def test_resnet(tmp_path, show_modules):
    model = models.resnet18(weights=None).eval()

    input_tensor = torch.randn(1, 3, 224, 224)

    with torchtrail.trace():
        input_tensor = torch.as_tensor(input_tensor)
        output_tensor = model(input_tensor)
        output_tensor = torch.nn.functional.softmax(output_tensor[0], dim=0)

    torchtrail.visualize(
        output_tensor, show_modules=show_modules, file_name=tmp_path / "resnet18.svg"
    )
    assert len(torchtrail.get_graph(output_tensor)) == 4
    assert len(torchtrail.get_graph(output_tensor, flatten=True)) == 174

    codegen_output = torchtrail.codegen(output_tensor)
    assert len(codegen_output.split("\n")) == 504
