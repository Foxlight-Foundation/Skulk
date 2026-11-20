# Native music tensor prefix evidence

These three original Safetensors prefixes contain **154,496 bytes** total.
They include the eight-byte length word and original JSON/padding, with no
tensor-body bytes. They were retained with the registry tensor-layout contract;
`layouts.json` is the corresponding exact grouped projection.

| Role | Selected repository / revision | File | Complete size | Prefix bytes |
| --- | --- | --- | ---: | ---: |
| Base | `ACE-Step/acestep-v15-base` / `e432212fec32b8965a14ffa57ae653438d6abd14` | `model.safetensors` | 4,787,825,604 | 80,568 |
| Text encoder | `ACE-Step/Ace-Step1.5` / `19671f406d603126926c1b7e2adc169acbcade22` | `Qwen3-Embedding-0.6B/model.safetensors` | 1,191,586,416 | 33,392 |
| Autoencoder | Same pinned companion source | `vae/diffusion_pytorch_model.safetensors` | 337,431,388 | 40,536 |

Complete object SHA-256 pins and MIT attribution remain in the [card fixture](../ace-native-card.toml). Tests verify the actual retained prefix digests,
complete scalar counts and equality with the registry projection. These facts
do not verify the absent tensor bodies, qualify model execution, establish peak
RAM requirements or grant hardware permission. Production independently verifies
the complete selected files before using their observed headers.
