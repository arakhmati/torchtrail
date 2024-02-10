import pytest

import urllib

import torch
from PIL import Image
from torchvision import transforms

import torchtrail


@pytest.mark.parametrize("show_modules", [True, False])
def test_resnet(tmp_path, show_modules):
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).eval()

    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    filename = tmp_path / "dog.jpg"

    try:
        urllib.URLopener().retrieve(url, filename)
    except Exception:
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image).unsqueeze(0)

    with torchtrail.trace():
        input_tensor = torch.as_tensor(input_tensor)
        output_tensor = model(input_tensor)
        output_tensor = torch.nn.functional.softmax(output_tensor[0], dim=0)

    torchtrail.visualize(
        output_tensor, show_modules=show_modules, file_name=tmp_path / "resnet18.svg"
    )
    assert len(torchtrail.get_graph(output_tensor)) == 4
    assert len(torchtrail.get_graph(output_tensor, flatten=True)) == 174
