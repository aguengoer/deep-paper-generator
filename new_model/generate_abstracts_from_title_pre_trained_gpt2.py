import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from traning_model_with_gpt_2_for_abstract import GPT2Model  # Import the GPT2Model class from your training script
from torchtext.data.metrics import bleu_score
from lime.lime_text import LimeTextExplainer
import json
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config


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


def evaluate_model(model, test_data):
    bleu_scores = []
    lime_scores = []
    explainer = LimeTextExplainer()

    for data in tqdm(test_data):
        title = data["title"]
        real_abstract = data["abstract"]

        generated_abstract = generate_abstract(model, title)

        # Calculate BLEU score
        bleu_scores.append(bleu_score([generated_abstract.split()], [real_abstract.split()]))

        # Calculate LIME score
        def predict_proba(texts):
            model.eval()
            outputs = []
            for text in texts:
                generated_abstract = generate_abstract(model, text)
                bleu = bleu_score([generated_abstract.split()], [real_abstract.split()])
                outputs.append([1 - bleu, bleu])
            return torch.tensor(outputs).float().numpy()

        exp = explainer.explain_instance(title, predict_proba, num_features=5, num_samples=20)
        lime_scores.append(exp.score)

    return bleu_scores, lime_scores


def main():
    model_path = "best_model_gpt2.pth"
    model = GPT2Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    with open("test_data_20.json", "r") as f:
        test_data = json.load(f)

    bleu_scores, lime_scores = evaluate_model(model, test_data)

    print(f"Average BLEU score: {sum(bleu_scores) / len(bleu_scores)}")
    print(f"Average LIME score: {sum(lime_scores) / len(lime_scores)}")


if __name__ == "__main__":
    main()