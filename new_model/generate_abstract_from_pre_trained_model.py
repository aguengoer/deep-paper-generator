import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load pre-trained model tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def generate_abstract(title, model, tokenizer, max_len=200, num_return_sequences=1):
    # Prepare the input for the model
    input_ids = tokenizer.encode(title, return_tensors="pt")

    # Set the generation parameters
    gen_options = {
        "max_length": max_len,
        "num_return_sequences": num_return_sequences,
        "no_repeat_ngram_size": 2,
        "temperature": 0.7,
        "do_sample": True,
    }

    # Generate text
    with torch.no_grad():
        generated_output = model.generate(input_ids, **gen_options)

    # Decode the generated text
    generated_abstracts = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_output]

    return generated_abstracts


# Test the function
title = "Deep Learning with NLP"
generated_abstracts = generate_abstract(title, model, tokenizer)
print("Generated Abstract: ", generated_abstracts[0])
