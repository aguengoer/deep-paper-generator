import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from traning_model_with_gpt_2_for_abstract import GPT2Model  # Import the GPT2Model class from your training script

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def generate_abstract(model, title, max_length=100, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Encode the title and add the eos_token at the end
        input_str = f"{title}{tokenizer.eos_token}"
        input_ids = torch.tensor(tokenizer.encode(input_str), dtype=torch.long).unsqueeze(0).to(device)

        # Generate abstract tokens
        for _ in range(max_length):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=-1)

            # Stop generating when the second eos_token is encountered
            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_abstract = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)

        # Remove title from the generated_abstract
        generated_abstract = generated_abstract[len(title):]

        return generated_abstract.strip()


def main():
    model_path = "best_model_gpt2.pth"
    model = GPT2Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_title = "Deep learning for natural language processing"
    generated_abstract = generate_abstract(model, test_title)
    print(f"Title: {test_title}")
    print(f"Generated abstract: {generated_abstract}")


if __name__ == "__main__":
    main()
