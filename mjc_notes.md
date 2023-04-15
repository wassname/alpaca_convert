
My personal repo to convert models from Lora to huggingface/ggml/gptq 4bit so I can run them in normal text-webui and llama.cpp

How do we do this?

1. lora -> hf
    - [tloen/alpaca-lora/export_hf_checkpoint.py](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py)
2. hf -> 4bit
    - using [GPTQ-for-LLaMa/llama.py](https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/triton/llama.py)
    `CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save llama7b-4bit-128g.pt`
3) and to ggml
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

# 4bit ones if usefull?
python scripts/download-model.py decapoda-research/llama-7b-hf-int4
python scripts/download-model.py decapoda-research/llama-13b-hf-int4
python scripts/download-model.py decapoda-research/llama-30b-hf-int4
wget https://huggingface.co/maderix/llama-65b-4bit/resolve/main/llama30b-4bit.pt ./models/decapoda-research_llama-30b-hf-int4/llama-30b-4bit.pt
# because the last repo is mostly empty we will combine...
python scripts/download-model.py decapoda-research/llama-65b-hf-int4
wget https://huggingface.co/maderix/llama-65b-4bit/resolve/main/llama65b-4bit.pt ./models/decapoda-research_llama-65b-hf-int4/llama-65b-4bit.pt
cp models/decapoda-research_llama-7b-hf/*.json models/decapoda-research_llama-7b-hf-int4

# oh! you need to replace LLaMATokenizer with LlamaTokenizer in all model json files

# usefull?
# wget https://huggingface.co/maderix/llama-65b-4bit/resolve/main/llama30b-4bit.pt ../llama-30b-4bit.pt
# wget https://huggingface.co/maderix/llama-65b-4bit/resolve/main/llama13b-4bit.pt ../llama-13b-4bit.pt
# wget https://huggingface.co/maderix/llama-65b-4bit/resolve/main/llama7b-4bit.pt ../llama-7b-4bit.pt

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
python scripts/export_hf_checkpoint.py ./models/decapoda-research_llama-13b-hf -l loras/chansung_alpaca-lora-13b

# or from int4?
python -m pdb scripts/export_hf_checkpoint_int4.py ./models/decapoda-research_llama-7b-hf ./models/decapoda-research_llama-7b-hf-int4/llama-7b-4bit.pt -l ./loras/tloen_alpaca-lora-7b



python scripts/export_hf_checkpoint.py ./models/llama-7b-hf -l loras/tloen_alpaca-lora-7b
python scripts/export_hf_checkpoint.py ./models/llama-13b-hf -l loras/chansung_alpaca-lora-13b # crash! 50GB+ needed
python scripts/export_hf_checkpoint.py ./models/llama-30b-hf -l loras/chansung_alpaca-lora-30b
python scripts/export_hf_checkpoint.py ./models/llama-60b-hf -l loras/chansung_alpaca-lora-60b
# test
python scripts/test_01_delora.py models/tloen_alpaca-lora-7b-delorified
```



# Links

- https://github.com/s4rduk4r/alpaca_lora_4bit_readme/blob/main/README.md


# 2023-04-13 16:44:11

OK I need lots more mem... copy to ec2

```sh
rsync -a . alpaca:/home/ubuntu/alpaca_convert_mjc --exclude=models
```
