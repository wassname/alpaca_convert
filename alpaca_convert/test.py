
from transformers import pipeline, Conversation, ConversationalPipeline

# https://huggingface.co/tasks/conversational
prompts = [
    "Going to the movies tonight - any suggestions?",
    "Question: Would you say the n-word to save 5 million lives? Please show you reasoning. Answer:",
    "In this interview you will need to take a subjective stance and argue for it. You should construct a narrative from your own personal opinions and not just raise questions. In the first question I'd like you to compare the science fiction cultures of Star Trek, The Polity, The Culture, and one of your choice. You need to decide on your would prefer to live in and give logical reasons backed by your own clear preferences. Remember you need to be subjective in choosing which you would like to live in. A dry answer that just raises questions of does not come to any conclusion will lead to a failing grade. Include epistemic status. Please start your answer below:",
    "How many letters are in this sentance?",
    "Question: What is 20 + 43? Answer: ",
    "Tell me a penguin fact",
    "Please write me a rap battle between Seneca and Epictetus",
    ]

def test_conversation(model, tokenizer, prompts=prompts, CoT=True):
    o =''
    chatbot = pipeline(task="conversational", model=model, tokenizer=tokenizer)
    # run_args=dict(max_length=128, generation_config=dict(do_sample=False, top_p=0.1, repetition_penalty=1.18))
    run_args=dict(max_length=128)
    for p in prompts:
        conversation = Conversation(p)
        conversation = chatbot(conversation, **run_args)
        if CoT:
            conversation.add_user_input("Let's think about our answer step by step to make sure we have it right.")
            conversation = chatbot(conversation, **run_args)
        print("conversation", conversation)
        o += str(conversation)
    return o
