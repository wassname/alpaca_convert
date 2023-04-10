
My personal repo to convert models from Lora to huggingface/ggml/gptq 4bit so I can run them in normal text-webui and llama.cpp

How do we do this?

1. lora -> hf
    - [tloen/alpaca-lora/export_hf_checkpoint.py](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py)
2. hf -> 4bit
    - using [GPTQ-for-LLaMa/llama.py](https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/triton/llama.py)
    `CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save llama7b-4bit-128g.pt`
3. 4bit -> ggml
    - [llama.cpp/convert-pth-to-ggml.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-gptq-to-ggml.py)


# TODO

- [x] lora -> hf
    - [ ] test this
- [ ] hf -> 4bit
- [ ] 4bit to -> ggml

# setup env

```sh

conda create -n textgen3 python=3.10.9
conda activate textgen3
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit-dev==11.7  cudatoolkit=11.7 -c pytorch -c nvidia  -c conda-forge 
```

# download models

```sh
# # base models.... FIXME


# download loras
python scripts/download-model.py chansung/alpaca-lora-30b
python scripts/download-model.py chansung/alpaca-lora-13b
python scripts/download-model.py tloen/alpaca-lora-7b
```

# convert models

```sh
python scripts/export_hf_checkpoint.py ./models/llama-7b-hf -l loras/tloen_alpaca-lora-7b
```


# Links

- https://github.com/s4rduk4r/alpaca_lora_4bit_readme/blob/main/README.md
