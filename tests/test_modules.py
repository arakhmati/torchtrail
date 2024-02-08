import pytest

import torch
import transformers

import torchtrail


@pytest.mark.parametrize("show_modules", [True, False])
def test_bert(tmp_path, show_modules):
    model_name = "google/bert_uncased_L-4_H-256_A-4"
    model = transformers.BertModel.from_pretrained(model_name).eval()

    with torchtrail.trace():
        input_tensor = torch.randint(0, model.config.vocab_size, (1, 64))
        output_tensor = model(input_tensor)

    torchtrail.visualize(
        output_tensor, show_modules=show_modules, file_name=tmp_path / "bert.svg"
    )
    assert len(torchtrail.get_graph(output_tensor)) == 2
    assert len(torchtrail.get_graph(output_tensor, flatten=True)) == 205


@pytest.mark.parametrize("show_modules", [True, False])
def test_resnet(tmp_path, show_modules):
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).eval()

    import urllib

    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    filename = tmp_path / "dog.jpg"

    try:
        urllib.URLopener().retrieve(url, filename)
    except Exception:
        urllib.request.urlretrieve(url, filename)

    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms

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


def test_module_list():

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Conv2d(3, 3, 3)] * 4)

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
