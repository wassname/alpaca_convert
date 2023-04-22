
My personal repo to convert models from Lora to huggingface/ggml/gptq 4bit so I can run them in normal text-webui and llama.cpp

How do we do this?

1. lora -> hf
    - [tloen/alpaca-lora/export_hf_checkpoint.py](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py)
2. hf -> 4bit
    - using [GPTQ-for-LLaMa/llama.py](https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/triton/llama.py)
    `CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save llama7b-4bit-128g.pt`
3. and to ggml
    - [llama.cpp/convert-pth-to-ggml.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-pth-to-ggml.py)


# TODO

- [ ] lora -> hf
- [ ] hf -> 4bit
- [ ] hf -> ggml

# setup env

```sh

conda create -n textgen4 python=3.10.9 -y
conda activate textgen4
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit-dev==11.7  cudatoolkit=11.7 -c pytorch -c nvidia  -c conda-forge  -y
pip install -r requirements.txt
pip install git+https://github.com/sterlind/GPTQ-for-LLaMa.git@lora_4bit
```

# download models

```sh
# # base models.... FIXME
python scripts/download-model.py decapoda-research/llama-7b-hf
python scripts/download-model.py decapoda-research/llama-13b-hf
python scripts/download-model.py decapoda-research/llama-30b-hf
python scripts/download-model.py decapoda-research/llama-65b-hf
# oh! you need to replace LLaMATokenizer with LlamaTokenizer in all model json files

# download loras
python scripts/download-model.py chansung/alpaca-lora-30b
python scripts/download-model.py chansung/alpaca-lora-13b
python scripts/download-model.py tloen/alpaca-lora-7b
python scripts/download-model.py gpt4all-alpaca-oa-codealpaca-lora-13b
python scripts/download-model.py Black-Engineer/oasst-llama30b-ggml-q4
```

# convert models

```sh
# download
python scripts/download-model.py tloen/alpaca-lora-7b
python scripts/download-model.py decapoda-research/llama-7b-hf

# convert
python scripts/01_export_hf_checkpoint.py ./data/models/decapoda-research_llama-7b-hf -l ./data/loras/tloen_alpaca-lora-7b
python scripts/01_export_hf_checkpoint.py ./data/models/decapoda-research_llama-13b-hf -l ./data/loras/chansung_alpaca-lora-13b # crash! 50GB+ needed
python scripts/01_export_hf_checkpoint.py ./data/models/decapoda-research_llama-30b-hf -l ./data/loras/chansung_alpaca-lora-30b
python scripts/01_export_hf_checkpoint.py ./data/models/decapoda-research_llama-60b-hf -l ./data/loras/chansung_alpaca-lora-60b

# test
python scripts/test_01_delora.py models/tloen_alpaca-lora-7b-delorified
python scripts/test_01_delora.py models/chansung_alpaca-lora-13b-delorified
# now compare what was generated during conversion `test_prompts.txt`, to the loaded version

# 4bit...
CUDA_VISIBLE_DEVICES=0 python llama.py ./data/models/tloen_alpaca-lora-7b-delorified c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save_safetensors ./data/models/tloen_alpaca-lora-7b-delorified/llama7b-4bit-128g.safetensors

# ggml conversion...
```



# Links

- https://github.com/s4rduk4r/alpaca_lora_4bit_readme/blob/main/README.md


# 2023-04-13 16:44:11

OK I need lots more mem... copy to ec2

```sh
rsync -a . alpaca:/home/ubuntu/alpaca_convert_mjc --exclude=models
```
