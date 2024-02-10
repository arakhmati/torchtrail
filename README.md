# torchtrail

[![PyPI version](https://badge.fury.io/py/torchtrail.svg)](https://badge.fury.io/py/torchtrail)
[![Build Status](https://github.com/arakhmati/torchtrail/actions/workflows/python-package.yml/badge.svg)](https://github.com/arakhmati/torchtrail/actions/workflows/python-package.yml)
[![GitHub license](https://img.shields.io/github/license/arakhmati/torchtrail)](https://github.com/arakhmati/torchtrail/blob/main/LICENSE)

`torchtrail` provides an external API to trace pytorch models and extract the graph of torch functions and modules that were executed. The graphs can then be visualized or used for other purposes.

## Installation Instructions

### On MacOs
```bash
brew install graphviz
pip install torchtrail
```

### On Ubuntu
```bash
sudo apt-get install graphviz
pip install torchtrail
```

## Examples

### Tracing a function
```python
import torch
import torchtrail

with torchtrail.trace():
    input_tensor = torch.rand(1, 64)
    output_tensor = torch.exp(input_tensor)
torchtrail.visualize(output_tensor, file_name="exp.svg")
```
![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/exp.svg)

The graph could be obtained as a `networkx.MultiDiGraph` using `torchtrail.get_graph`:
```python
graph: "networkx.MultiDiGraph" = torchtrail.get_graph(output_tensor, as_networkx=True)
```


### Tracing a module

```python
import torch
import transformers

import torchtrail

model_name = "google/bert_uncased_L-4_H-256_A-4"
config = transformers.BertConfig.from_pretrained(model_name)
config.num_hidden_layers = 1
model = transformers.BertModel.from_pretrained(model_name, config=config).eval()

with torchtrail.trace():
    input_tensor = torch.randint(0, model.config.vocab_size, (1, 64))
    output = model(input_tensor).last_hidden_state

torchtrail.visualize(output, max_depth=1, file_name="bert_max_depth_1.svg")
```

![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/bert_max_depth_1.svg)


```python
torchtrail.visualize(output, max_depth=2, file_name="bert_max_depth_2.svg")
```

![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/bert_max_depth_2.svg)

The graph of the full module can be visualized by omitting `max_depth` argument

```python
torchtrail.visualize(output, file_name="bert.svg")
```

![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/bert.svg)

The graph could be obtained as a `networkx.MultiDiGraph` using `torchtrail.get_graph`:
```python
graph: "networkx.MultiDiGraph" = torchtrail.get_graph(output_tensor, as_networkx=True)
```

Alternatively, visualization of the modules can be turned off completely using `show_modules=False`

```python
torchtrail.visualize(output, show_modules=False, file_name="bert_show_modules_False.svg")
```

![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/bert_show_modules_False.svg)

The flattened graph could be obtained as a `networkx.MultiDiGraph` using `torchtrail.get_graph`:
```python
graph: "networkx.MultiDiGraph" = torchtrail.get_graph(output_tensor, as_networkx=True, flatten=True)
```


## Reference
- `torchtrail` was inspired by [torchview](https://github.com/mert-kurttutan/torchview). [mert-kurttutan](https://github.com/mert-kurttutan) did an amazing job with displaying torch graphs. However, one of the goals of `torchtrail` included producing [networkx](https://networkx.org)-compatible graph, therefore `torchtrail` was written.
- The idea to use persistent MultiDiGraph to trace torch operations was taken from [composit](https://github.com/arakhmati/composit)
