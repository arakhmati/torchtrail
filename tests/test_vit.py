import pytest

import torch
import transformers

import torchtrail


@pytest.mark.parametrize("show_modules", [True, False])
def test_vit_embeddings(tmp_path, show_modules):
    model_name = "google/vit-base-patch16-224"
    batch_size = 1
    num_channels = 3
    height = 224
    width = 224

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTEmbeddings(config).eval()

    with torchtrail.trace():
        input_tensor = torch.randn(batch_size, num_channels, height, width)
        output = model(input_tensor)

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "vit_embeddings.svg"
    )


@pytest.mark.parametrize("show_modules", [True, False])
def test_vit_self_attention(tmp_path, show_modules):
    model_name = "google/vit-base-patch16-224"
    batch_size = 1
    sequence_size = 197
    hidden_size = 768

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTSelfAttention(config).eval()

    with torchtrail.trace():
        input_tensor = torch.randn(batch_size, sequence_size, hidden_size)
        output = model(input_tensor)

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "vit_self_attention.svg"
    )


@pytest.mark.parametrize("show_modules", [True, False])
def test_vit_self_output(tmp_path, show_modules):
    model_name = "google/vit-base-patch16-224"
    batch_size = 1
    sequence_size = 197
    hidden_size = 768

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTSelfOutput(config).eval()

    with torchtrail.trace():
        input_tensor = torch.randn(batch_size, sequence_size, hidden_size)
        residual_input_tensor = torch.randn(batch_size, sequence_size, hidden_size)
        output = model(input_tensor, residual_input_tensor)

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "vit_self_output.svg"
    )


@pytest.mark.parametrize("show_modules", [True, False])
def test_vit_attention(tmp_path, show_modules):
    model_name = "google/vit-base-patch16-224"
    batch_size = 1
    sequence_size = 197
    hidden_size = 768

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTAttention(config).eval()

    with torchtrail.trace():
        input_tensor = torch.randn(batch_size, sequence_size, hidden_size)
        output = model(input_tensor)

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "vit_attention.svg"
    )


@pytest.mark.parametrize("show_modules", [True, False])
def test_vit_intermediate(tmp_path, show_modules):
    model_name = "google/vit-base-patch16-224"
    batch_size = 1
    sequence_size = 197
    hidden_size = 768

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTIntermediate(config).eval()

    with torchtrail.trace():
        input_tensor = torch.randn(batch_size, sequence_size, hidden_size)
        output = model(input_tensor)

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "vit_intermediate.svg"
    )


@pytest.mark.parametrize("show_modules", [True, False])
def test_vit_layer(tmp_path, show_modules):
    model_name = "google/vit-base-patch16-224"
    batch_size = 1
    sequence_size = 197
    hidden_size = 768

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTLayer(config).eval()

    with torchtrail.trace():
        input_tensor = torch.randn(batch_size, sequence_size, hidden_size)
        output = model(input_tensor)

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "vit_layer.svg"
    )


@pytest.mark.parametrize("show_modules", [True, False])
def test_vit_encoder(tmp_path, show_modules):
    model_name = "google/vit-base-patch16-224"
    batch_size = 1
    sequence_size = 197
    hidden_size = 768

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTEncoder(config).eval()

    with torchtrail.trace():
        input_tensor = torch.randn(batch_size, sequence_size, hidden_size)
        output = model(input_tensor).last_hidden_state

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "vit_encoder.svg"
    )


@pytest.mark.parametrize("show_modules", [True, False])
def test_vit(tmp_path, show_modules):
    model_name = "google/vit-base-patch16-224"
    batch_size = 1
    num_channels = 3
    height = 224
    width = 224

    config = transformers.ViTConfig.from_pretrained(model_name)
    model = transformers.models.vit.modeling_vit.ViTModel(config).eval()

    with torchtrail.trace():
        input_tensor = torch.randn(batch_size, num_channels, height, width)
        output = model(input_tensor).pooler_output

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "vit.svg"
    )
