import pytest

import torch
import transformers

import torchtrail


@pytest.mark.parametrize("trace_modules", [True, False])
def test_bert(tmp_path, trace_modules):
    model_name = "google/bert_uncased_L-4_H-256_A-4"
    model = transformers.BertModel.from_pretrained(model_name).eval()

    with torchtrail.trace(trace_modules=trace_modules):
        input_tensor = torch.randint(0, model.config.vocab_size, (1, 64))
        output = model(input_tensor).last_hidden_state

    if trace_modules:
        assert len(output.graph) == 2
    else:
        assert len(output.graph) == 196

    torchtrail.visualize(output, file_name=tmp_path / "bert.dot")


@pytest.mark.parametrize("trace_modules", [True, False])
def test_resnet(tmp_path, trace_modules):
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).eval()

    import urllib

    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    filename = tmp_path / "dog.jpg"

    try:
        urllib.URLopener().retrieve(url, filename)
    except:
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

    with torchtrail.trace(trace_modules=trace_modules):
        input_tensor = torch.as_tensor(input_tensor)
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    if trace_modules:
        assert len(probabilities.graph) == 4
    else:
        assert len(probabilities.graph) == 174

    torchtrail.visualize(probabilities, file_name=tmp_path / "resnet18.dot")


@pytest.mark.parametrize("trace_modules", [True, False])
def test_vit(tmp_path, trace_modules):
    model_name = "google/vit-base-patch16-224"
    batch_size = 1
    num_channels = 3
    height = 224
    width = 224

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTModel(config).eval()

    with torchtrail.trace(trace_modules=trace_modules):
        input_tensor = torch.randn(batch_size, num_channels, height, width)
        output = model(input_tensor).pooler_output

    torchtrail.visualize(
        output, file_name=tmp_path / "vit-base-patch-16-224.dot", max_depth=4
    )
