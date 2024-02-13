import pytest

import os

import torch
import transformers

import torchtrail


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.parametrize("show_modules", [True, False])
def test_falcon_decoder_layer(tmp_path, show_modules):

    model_name = "tiiuae/falcon-7b-instruct"
    config = transformers.FalconConfig.from_pretrained(model_name)
    model = transformers.models.falcon.modeling_falcon.FalconDecoderLayer(config)

    with torchtrail.trace():
        input_tensor = torch.rand((1, 64, 4544))
        attention_mask = torch.ones((1, 64), dtype=torch.long)
        output = model(input_tensor, alibi=None, attention_mask=attention_mask)

    torchtrail.visualize(
        output,
        show_modules=show_modules,
        file_name=tmp_path / "falcon_decoder_layer.svg",
    )
    assert len(torchtrail.get_graph(output)) == 3
    assert len(torchtrail.get_graph(output, flatten=True)) == 56


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.mark.parametrize("show_modules", [True, False])
def test_falcon(tmp_path, show_modules):

    model_name = "tiiuae/falcon-7b-instruct"
    model = (
        transformers.models.falcon.modeling_falcon.FalconForCausalLM.from_pretrained(
            model_name
        )
    )

    with torchtrail.trace():
        input_tensor = torch.randint(0, model.config.vocab_size, (1, 64))
        output = model(input_tensor)

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "falcon.svg"
    )
    assert len(torchtrail.get_graph(output)) == 2
    assert len(torchtrail.get_graph(output, flatten=True)) == 1578
