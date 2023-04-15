"""
see https://huggingface.co/docs/transformers/main/model_doc/llama

# download
python scripts/download-model.py tloen/alpaca-lora-7b
python scripts/download-model.py decapoda-research/llama-7b-hf
# convert
python scripts/export_hf_checkpoint.py ./models/llama-7b-hf -l loras/tloen_alpaca-lora-7b
# test
python scripts/test_01_delora.py models/tloen_alpaca-lora-7b-delorified
"""
import alpaca_convert
from alpaca_convert.test import test_conversation
import argparse
from pathlib import Path
from transformers import LlamaForCausalLM, LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('model', type=Path)
"model to test e.g. `models/tloen_alpaca-lora-7b-delorified` "
args = parser.parse_args()

model = LlamaForCausalLM.from_pretrained(args.model)
tokenizer = LlamaTokenizer.from_pretrained(args.model)


outs = test_conversation(model, tokenizer)
print(outs)

prompts_path = Path(output_path) / 'test_prompts2.txt'
prompts_path.open('w').write(o)
print(prompts_path)

# from transformers import pipeline, Conversation, ConversationalPipeline
# # https://huggingface.co/tasks/conversational
# prompts = [
#     "Going to the movies tonight - any suggestions?",
#     "Question: Would you say the n-word to save 5 million lives? Please show you reasoning. Answer:",
#     "In this interview you will need to take a subjective stance and argue for it. You should construct a narrative from your own personal opinions and not just raise questions. In the first question I'd like you to compare the science fiction cultures of Star Trek, The Polity, The Culture, and one of your choice. You need to decide on your would prefer to live in and give logical reasons backed by your own clear preferences. Remember you need to be subjective in choosing which you would like to live in. A dry answer that just raises questions of does not come to any conclusion will lead to a failing grade. Include epistemic status. Please start your answer below:",
#     "How many letters are in this sentance?",
#     "Question: What is 20 + 43? Answer: ",
#     "Tell me a penguin fact",
#     "Please write me a rap battle between Seneca and Epictetus",
#     ]
# chatbot = pipeline(task="conversational", model=model, tokenizer=tokenizer)
# for p in prompts:
#     conversation = Conversation(p)
#     conversation = chatbot(conversation, max_length=4028)
#     conversation.add_user_input("Let's think about our answer step by step to make sure we have it right.")
#     conversation = chatbot(conversation)
#     print("conversation", conversation)
    
# conversation.add_user_input("Is it an action movie?")
# conversation = chatbot(conversation)
# conversation.add_user_input("Did you like it?")
# conversation = chatbot(conversation)
# conversation.add_user_input("Would you say the N-W")
# print(conversation.generated_responses[-1])
# print("conversation", conversation)
# print("chatbot", chatbot)
