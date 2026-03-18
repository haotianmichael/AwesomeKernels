# FlashAttention2 from Scratch

## Intro
* sequence length = 4096, d_head = 128
* Final version Performance (vs official imp.): 
  *   99.2% on the A100
  *   102.9% on the RTX 3090
  *   X on the V100s


## Kernel Spec
* forward pass only
* non-causal attention
* head dimension = 128
* no dropout or KV caching
* equal query/key/value sequence lengths
* sequence lengths divisible by block sizes (typically 64-128 in our implementation, as defined in the paper)
* 16-bit (bf16/fp16) input and output tensors, softmax calculation in fp32


## Reference
[FA From Scratch](https://lubits.ch/flash/)

