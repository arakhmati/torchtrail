import pytest

import torch
import transformers

import torchtrail


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


@pytest.mark.skip(reason="Test takes too long to run.")
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


@pytest.mark.parametrize("show_modules", [True, False])
@pytest.mark.parametrize("num_hidden_layers", [1, 2])
@pytest.mark.parametrize("num_tokens", [1, 3])
@pytest.mark.parametrize("use_cache", [True, False])
def test_falcon7b_instruct_with_kv_cache(
    tmp_path, show_modules, num_hidden_layers, num_tokens, use_cache
):

    model_version = "tiiuae/falcon-7b-instruct"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_version)

    config = transformers.models.falcon.modeling_falcon.FalconConfig.from_pretrained(
        model_version
    )
    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers

    model = transformers.models.falcon.modeling_falcon.FalconForCausalLM(
        config=config
    ).eval()

    prompt_text = ["Display a graph of the falcon model"]

    tokenized_inputs = tokenizer(
        prompt_text, padding=False, add_special_tokens=False, return_tensors="pt"
    )
    input_tokens = tokenized_inputs["input_ids"]

    def post_process(logits):
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        tokens = next_tokens[:, None]
        return tokens

    kv_cache = None
    with torchtrail.trace():
        tokens = input_tokens
        for _ in range(num_tokens):
            model_output = model(tokens, past_key_values=kv_cache, use_cache=use_cache)
            tokens = post_process(logits=model_output.logits)
            kv_cache = model_output.past_key_values
            assert hasattr(tokens, "graph")

    torchtrail.visualize(
        tokens,
        file_name=tmp_path / "falcon7b_instruct_with_kv_cache.svg",
        show_modules=show_modules,
    )

    if num_hidden_layers == 1 and num_tokens == 1 and use_cache:
        assert len(torchtrail.get_graph(tokens)) == 5
        assert len(torchtrail.get_graph(tokens, flatten=True)) == 62
    elif num_hidden_layers == 1 and num_tokens == 1 and not use_cache:
        assert len(torchtrail.get_graph(tokens)) == 5
        assert len(torchtrail.get_graph(tokens, flatten=True)) == 62
    elif num_hidden_layers == 1 and num_tokens == 3 and use_cache:
        assert len(torchtrail.get_graph(tokens)) == 13
        assert len(torchtrail.get_graph(tokens, flatten=True)) == 188
    elif num_hidden_layers == 1 and num_tokens == 3 and not use_cache:
        assert len(torchtrail.get_graph(tokens)) == 13
        assert len(torchtrail.get_graph(tokens, flatten=True)) == 184
    elif num_hidden_layers == 2 and num_tokens == 1 and use_cache:
        assert len(torchtrail.get_graph(tokens)) == 5
        assert len(torchtrail.get_graph(tokens, flatten=True)) == 111
    elif num_hidden_layers == 2 and num_tokens == 1 and not use_cache:
        assert len(torchtrail.get_graph(tokens)) == 5
        assert len(torchtrail.get_graph(tokens, flatten=True)) == 111
    elif num_hidden_layers == 2 and num_tokens == 3 and use_cache:
        assert len(torchtrail.get_graph(tokens)) == 13
        assert len(torchtrail.get_graph(tokens, flatten=True)) == 339
    elif num_hidden_layers == 2 and num_tokens == 3 and not use_cache:
        assert len(torchtrail.get_graph(tokens)) == 13
        assert len(torchtrail.get_graph(tokens, flatten=True)) == 331
    else:
        raise ValueError(
            f"Unexpected combination of num_hidden_layers={num_hidden_layers}, num_tokens={num_tokens}, use_cache={use_cache}"
        )
