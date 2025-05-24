import pytest

import torch
import transformers

import torchtrail


@pytest.mark.parametrize("show_modules", [True, False])
def test_bert_attention(tmp_path, show_modules):
    config = transformers.BertConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
    )
    model = transformers.models.bert.modeling_bert.BertAttention(config).eval()

    with torchtrail.trace():
        input_tensor = torch.rand(1, 64, config.hidden_size)
        output = model(input_tensor)

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "bert_attention.svg"
    )
    assert len(torchtrail.get_graph(output)) == 2
    assert len(torchtrail.get_graph(output, flatten=True)) == 33

    codegen_output = torchtrail.codegen(output)
    assert len(codegen_output.split("\n")) == 77


@pytest.mark.parametrize("show_modules", [True, False])
def test_bert(tmp_path, show_modules):
    config = transformers.BertConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
    )
    model = transformers.BertModel(config).eval()

    with torchtrail.trace():
        input_tensor = torch.randint(0, model.config.vocab_size, (1, 64))
        output = model(input_tensor)

    torchtrail.visualize(
        output, show_modules=show_modules, file_name=tmp_path / "bert.svg"
    )
    assert len(torchtrail.get_graph(output)) == 2
    assert len(torchtrail.get_graph(output, flatten=True)) == 206

    codegen_output = torchtrail.codegen(output)
    assert len(codegen_output.split("\n")) == 542
