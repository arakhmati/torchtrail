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
        output = model(input_tensor).last_hidden_state

    assert len(output.graph) == 2
    if not show_modules:
        assert len(torchtrail.flatten_graph(output.graph)) == 205

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "bert.svg"
    )


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
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    assert len(output.graph) == 2
    if not show_modules:
        assert len(torchtrail.flatten_graph(output.graph)) == 172

    torchtrail.visualize(
        probabilities, show_modules=show_modules, file_name=tmp_path / "resnet18.svg"
    )
