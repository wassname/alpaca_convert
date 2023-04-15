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
import autograd_4bit
from autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear

def main(BASE_MODEL, LORA_MODEL, int4_checkpoint_path, output_path=None):
    
    if output_path is None:
        output_path = 'models/' + LORA_MODEL.split('/')[-1] + '-delorified'

    # load 4bit, from https://github.com/johnsmith0031/alpaca_lora_4bit/blob/fb7665726e5b69dcac6020707bbece7b0d39b865/text-generation-webui/custom_monkey_patch.py#L4
    model, tokenizer = load_llama_model_4bit_low_ram(config_path=BASE_MODEL, model_path=int4_checkpoint_path, groupsize=-1, is_v1_model=True)
    lora_model = PeftModel.from_pretrained(model, LORA_MODEL, device_map={'': "cpu"}, torch_dtype=torch.float16)
    print('{} Lora Applied.'.format(lora_path))
    
    print('Apply auto switch and half')
    for n, m in lora_model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
            if m.is_v1_model:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()
            m.bias = m.bias.half()
    autograd_4bit.use_new = True
    autograd_4bit.auto_switch = True

    # tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

    # base_model = LlamaForCausalLM.from_pretrained(
    #     BASE_MODEL,
    #     load_in_8bit=False,
    #     torch_dtype=torch.float16,
    #     device_map={"": "cpu"},
    # )
    
    # # TODO or load 4 bit?

    # first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    # first_weight_old = first_weight.clone()

    # lora_model = PeftModel.from_pretrained(
    #     base_model,
    #     LORA_MODEL,
    #     device_map={"": "cpu"},
    #     torch_dtype=torch.float16,
    # )

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
    LlamaTokenizer.save_pretrained(tokenizer, output_path)
    # FIXME also save tokenizer
    
    from alpaca_convert.test import test_conversation
    o = test_conversation(lora_model.float(), tokenizer)
    print(o)
    prompts_path = Path(output_path) / 'test_prompts.txt'
    print(prompts_path)
    prompts_path.open('w').write(o)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('int4_checkpoint_path', type=str)
    parser.add_argument('-l', '--lora', type=str, default='main', help='Lora repo or path e.g. `tloen/alpaca-lora-7b`')
    parser.add_argument('-o', '--output', type=Path, default=None)
    "e.g. ./hf_ckpt. default will be lora name"
    args = parser.parse_args()
    print(args)
    main(args.model, args.lora, args.int4_checkpoint_path, args.output)

