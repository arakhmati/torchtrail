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
    tensor = torch.rand(1, 64)
    tensor = torch.exp(tensor)
torchtrail.visualize(tensor)
```
![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/exp.svg)


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

torchtrail.visualize(output)
```

![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/bert.svg)

The graph of the module can be visualized differently using `max_depth` argument

```python
torchtrail.visualize(output, max_depth=1)
```

![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/bert_max_depth_1.svg)

```python
torchtrail.visualize(output, max_depth=2)
```

![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/bert_max_depth_2.svg)

Alternatively, visualization of the modules can be turned off using `show_modules=False`

```python
torchtrail.visualize(output, show_modules=False)
```

![](https://raw.githubusercontent.com/arakhmati/torchtrail/main/docs/images/bert_show_modules_False.svg)