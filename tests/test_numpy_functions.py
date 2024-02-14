import numpy as np

import torchtrail


def test_ones_exp(tmp_path):
    with torchtrail.trace():
        input_tensor = np.ones((1, 64))
        output_tensor = np.exp(input_tensor)

    torchtrail.visualize(output_tensor, file_name=tmp_path / "ones_exp.svg")
    assert len(torchtrail.get_graph(output_tensor)) == 2


def test_ones_split_add(tmp_path):
    with torchtrail.trace():
        input_tensor = np.ones((1, 64))
        output_tensors = np.split(input_tensor, indices_or_sections=2, axis=1)
        output_tensor = output_tensors[0] + output_tensors[1]

    torchtrail.visualize(output_tensor, file_name=tmp_path / "ones_split_add.svg")
    assert len(torchtrail.get_graph(output_tensor)) == 3
