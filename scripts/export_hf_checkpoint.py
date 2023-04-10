"""
From https://raw.githubusercontent.com/tloen/alpaca-lora/main/export_hf_checkpoint.py
"""
import os
from pathlib import Path
import argparse
import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

def main(BASE_MODEL, LORA_MODEL, output_path=None):
    
    if output_path is None:
        output_path = 'models/' + LORA_MODEL.split('/')[-1] + '-delorified'

    # BASE_MODEL = os.environ.get("BASE_MODEL", None)
    # assert (
    #     BASE_MODEL
    # ), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501


    # LORA_MODEL = os.environ.get("BASE_MODEL", None)
    # assert (
    #     LORA_MODEL
    # ), "Please specify a value for LORA_MODEL environment variable, e.g. `export BASE_MODEL=tloen/alpaca-lora-7b`"  # noqa: E501

    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.model.layers[
        0
    ].self_attn.q_proj.weight

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()

    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    LlamaForCausalLM.save_pretrained(
        base_model, output_path, state_dict=deloreanized_sd, max_shard_size="400MB"
    )
    print(f'output {output_path}')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('-l', '--lora', type=str, default='main', help='Lora repo or path e.g. `tloen/alpaca-lora-7b`')
    parser.add_argument('-o', '--output', type=Path, default=None)
    "e.g. ./hf_ckpt. default will be lora name"
    args = parser.parse_args()
    main(args.model, args.lora, args.output)

