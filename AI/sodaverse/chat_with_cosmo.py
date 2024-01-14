import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("allenai/cosmo-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/cosmo-xl").to(device)

def set_input(situation_narrative, role_instruction, conversation_history):
    input_text = " <turn> ".join(conversation_history)

    if role_instruction != "":
        input_text = "{} <sep> {}".format(role_instruction, input_text)

    if situation_narrative != "":
        input_text = "{} <sep> {}".format(situation_narrative, input_text)

    return input_text

def generate(situation_narrative, role_instruction, conversation_history):
    """
    situation_narrative: the description of situation/context with the characters included (e.g., "David goes to an amusement park")
    role_instruction: the perspective/speaker instruction (e.g., "Imagine you are David and speak to his friend Sarah").
    conversation_history: the previous utterances in the conversation in a list
    """

    input_text = set_input(situation_narrative, role_instruction, conversation_history)

    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=128, temperature=1.0, top_p=.95, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return response

situation = "Cosmo had a really fun time participating in the EMNLP conference at Abu Dhabi."
instruction = "You are Cosmo and you are talking to a friend." # You can also leave the instruction empty

conversation = [
    "Hey, how was your trip to Abu Dhabi?"
]

response = generate(situation, instruction, conversation)
print(response)