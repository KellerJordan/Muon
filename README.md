# Muon: An optimizer for the hidden layers of neural networks

This repo contains an implementation of the `Muon` optimizer originally described in [this thread](https://x.com/kellerjordan0/status/1842300916864844014) and [this writeup](https://kellerjordan.github.io/posts/muon/).

## Installation

```
pip install git+https://github.com/KellerJordan/Muon
```

## Usage

Muon is an optimizer for the hidden weights of a neural network.
Other parameters, such as embeddings, classifier heads, and hidden gains/biases should be optimized using standard AdamW.
Muon should be used as follows:

```python
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)

# To replace the above, do the following:

from muon import MuonWithAuxAdam
hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.02, weight_decay=0.01),
    dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
optimizer = MuonWithAuxAdam(param_groups)
```

You'll have to replace `model.body`, `model.head`, and `model.embed` with whatever is appropriate for your model.
E.g., for a ConvNet, you should use Muon to optimize all the convolutional filters except the first one, and AdamW to optimize everything else.

## Example usage

[Example use in the NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470)

[Example use in the CIFAR-10 speedrun](https://github.com/KellerJordan/cifar10-airbench/blob/28bff5f5b31e95aa45b5b20e1f48baf1ed98d5f6/airbench94_muon.py#L362)

## Hyperparameter tuning

Typically, the default values of momentum (0.95), nesterov (True), and ns_steps (5) work well. Only the learning rate and weight decay must be tuned.
The learning rate should have built-in muP scaling: That is, as you scale up the model size, you shouldn't need to retune it.

## Benchmarks

For a comparison between AdamW, Shampoo, SOAP, and Muon for training a 124M-parameter transformer, see [here](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/102924_Optimizers).

## Accomplishments

* [Lowered the record for training to 94% on CIFAR-10 from 3.3 A100-seconds to 2.6 A100-seconds](https://github.com/KellerJordan/cifar10-airbench)
* [Used to train a transformer to GPT-2 (XL) performance in $175 of compute](https://x.com/kellerjordan0/status/1850995958697308307)
* [Improved the training speed record for attaining GPT-2 (small) performance by a factor of 1.35x](https://x.com/kellerjordan0/status/1842300916864844014)
* [Used by the Kimi.ai frontier lab for scaled LLM training](https://x.com/Kimi_Moonshot/status/1893379158472044623)
* [Ashish Vaswani's lab essential.ai showed that Muon is especially good for training with large batch size](https://arxiv.org/abs/2505.02222)

## More learning resources and results about Muon

* [Blog post on Muon by Jianlin Su (the creator of RoPE)](https://kexue.fm/archives/10592)
* [Blog post by Jeremy Bernstein on theoretical background of Muon](https://jeremybernste.in/writing/deriving-muon)
* [Tech report by Kimi.ai on using Muon for scaled training](https://arxiv.org/abs/2502.16982v1)
* [Why we chose Muon: Our chain of thought (by Jianlin Su at Kimi.ai)](https://x.com/Kimi_Moonshot/status/1897929976948965870)

## Citation

```bibtex
@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                  Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://kellerjordan.github.io/posts/muon/}
}
```
