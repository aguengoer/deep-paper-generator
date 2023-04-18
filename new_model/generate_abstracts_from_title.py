import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from training_model_for_abstracts import TransformerModel, PaperDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocab(path):
    with open(path, "r") as f:
        vocab = json.load(f)
    vocab.itos = {int(k): v for k, v in vocab["itos"].items()}
    return vocab


def tokenize(text):
    return [token.lower() for token in text.split()]


def load_model(path):
    vocab = load_vocab("vocab.json")
    ntokens = len(vocab)
    emsize = 512
    nhead = 8
    nhid = 2048
    nlayers = 3
    dropout = 0.1

    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
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
