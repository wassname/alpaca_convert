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
    LlamaTokenizer.save_pretrained(tokenizer, output_path)
    print(f'output {output_path}')
    
    from alpaca_convert.test import test_conversation
    o = test_conversation(lora_model.float(), tokenizer)
    print(o)
    prompts_path = Path(output_path) / 'test_prompts.txt'
    prompts_path.open('w').write(o)
    print(prompts_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Huggingface model e.g. `./models/decapoda-research_llama-7b-hf`')
    parser.add_argument('-l', '--lora', type=str, default='main', help='Lora repo or path e.g. `tloen/alpaca-lora-7b` or `./loras/tloen_alpaca-lora-7b`')
    parser.add_argument('-o', '--output', type=Path, default=None)
    "if None default will be `./models/tloen_alpaca-lora-7b_delorafied` or similar"
    args = parser.parse_args()
    main(args.model, args.lora, args.output)

