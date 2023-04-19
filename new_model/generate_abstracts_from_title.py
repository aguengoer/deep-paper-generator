import json
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
from training_model_for_abstracts import \
    TransformerModel  # Replace 'your_training_script' with the name of your training script file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomVocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])


def rebuild_vocab(vocab_data):
    stoi = vocab_data["stoi"]
    itos = vocab_data["itos"]
    return CustomVocab(stoi, itos)


def generate_abstract(model, tokenizer, vocab, title, authors, categories, max_length=150):
    model.eval()
    with torch.no_grad():
        input_str = f"{title} {authors} {categories}"
        input_tokens = [vocab[token] for token in tokenizer(input_str)]
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).t().to(device)

        output_tokens = [vocab["<pad>"]]
        for _ in range(max_length):
            output_tensor = torch.tensor([output_tokens], dtype=torch.long).t().to(device)
            logits = model(input_tensor, output_tensor)
            next_token = torch.argmax(logits[-1, 0], dim=-1).item()
            if next_token == vocab["<pad>"]:
                break
            output_tokens.append(next_token)

        #print(" ".join([vocab.itos[token] for token in output_tokens[1:]]))
        return " ".join([vocab.itos[token] for token in output_tokens[1:]])


def main():
    model_path = "best_model.pth"
    vocab_path = "vocab.json"

    # Load the vocabulary
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
    vocab = rebuild_vocab(vocab_data)

    # Load the model
    ntokens = len(vocab)
    emsize = 512
    nhid = 2048
    nlayers = 6
    nhead = 8
    dropout = 0.1
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(model_path))

    # Define your inputs
    title = "Fermionic superstring loop amplitudes in the pure spinor formalism"
    authors = "Paul"
    categories = "math"

    abstract = generate_abstract(model, get_tokenizer('spacy', language='en_core_web_sm'), vocab, title, authors,
                                 categories)
    print("Generated Abstract:")
    print(abstract)


if __name__ == "__main__":
    main()
