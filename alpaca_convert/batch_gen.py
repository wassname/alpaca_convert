import torch
from transformers import GenerationConfig

def get_output_batch(
    model, tokenizer, prompts, generation_config=GenerationConfig(**{'temperature': 0.9, 'repetition_penalty': 1.2, 'do_sample': True, 'max_new_tokens': 256, 'use_cache': True, 'num_beams': 1, 'top_p': 0.9, 'top_k': 50})
):
    if len(prompts) == 1:
        encoding = tokenizer(prompts, return_tensors="pt")
        input_ids = encoding["input_ids"].cuda()
        generated_id = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=256
        )

        decoded = tokenizer.batch_decode(generated_id, skip_special_tokens=True)
        del input_ids, generated_id
        torch.cuda.empty_cache()
        return decoded
    else:
        encodings = tokenizer(prompts, padding=True, return_tensors="pt").to('cuda')
        generated_ids = model.generate(
            **encodings,
            generation_config=generation_config,
            max_new_tokens=256
        )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        del encodings, generated_ids
        torch.cuda.empty_cache()
        return decoded


def generate_prompt(prompt1):
    """The format for alpaca training.
    
    see: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L36 
    """
    context_string = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    return f"""{context_string}

    ### Input: {prompt1}

    ### Response: 
    """

def prompt_batch(
    model, tokenizer, instructions, gen_config
):

    instruct_prompts = [
        generate_prompt(instruct)
        for instruct in instructions
    ]
        
    bot_responses = get_output_batch(
        model, tokenizer, instruct_prompts, gen_config
    )
    return bot_responses
