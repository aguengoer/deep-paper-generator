import json
import torch
import torch.nn as nn
from training_model_for_abstracts import TransformerModel
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocab:
    def __init__(self):
        self.stoi = None
        self.itos = None


def load_vocab(path):
    with open(path, "r") as f:
        data = json.load(f)

    vocab = Vocab()
    vocab.stoi = data["stoi"]
    vocab.itos = data["itos"]

    return vocab


def tokenize(text):
    return [token.lower() for token in text.split()]


def load_model(path: str) -> Tuple[nn.Module, Vocab]:
    with open("vocab.json", "r") as f:
        data = json.load(f)
    vocab = Vocab()
    vocab.itos = data["itos"]
    vocab.stoi = data["stoi"]

    # Model Parameters
    ntokens = len(vocab.stoi)
    emsize = 512
    nhid = 2048
    nlayers = 6
    nhead = 8
    dropout = 0.1

    # Training Parameters
    batch_size = 16
    num_epochs = 10
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    # Load the model state_dict
    model_state_dict = torch.load(path)

    # Replace the model state_dict with the loaded state_dict
    model.load_state_dict(model_state_dict)

    return model, vocab


def generate_abstract(model, vocab, title, author, category, max_len=200):
    model.eval()
    src_text = f"{title} {author} {category}"
    src_tokens = tokenize(src_text)
    src_tensor = torch.tensor([vocab[token] for token in src_tokens], dtype=torch.long).unsqueeze(1).to(device)

    output_tokens = [vocab["<bos>"]]
    for i in range(max_len):
        tgt_tensor = torch.tensor(output_tokens, dtype=torch.long).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        next_token = output.argmax(2)[-1, 0].item()
        if next_token == vocab["<eos>"]:
            break
        output_tokens.append(next_token)

    abstract = " ".join(
        [vocab.itos[token] for token in output_tokens if token not in (vocab["<bos>"], vocab["<eos>"], vocab["<pad>"])])
    return abstract


def main():
    model_path = "best_model.pth"
    model, vocab = load_model(model_path)

    # Test the model with sample input data
    title = "Example Title"
    author = "Author Name"
    category = "Category"

    generated_abstract = generate_abstract(model, vocab, title, author, category)
    print("Generated Abstract:\n", generated_abstract)


if __name__ == "__main__":
    main()
