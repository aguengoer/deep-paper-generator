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


def generate_abstract(model, title, vocab, max_length=50, temperature=1.0):
    model.eval()

    tokenizer = get_tokenizer('basic_english')
    tokens = [vocab[token] for token in tokenizer(title)]

    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)
    abstract_tokens = []

    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_tensor, input_tensor)
            logits = output[-1, 0, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probabilities, 1).item()
            abstract_tokens.append(predicted_token)
            input_tensor = torch.cat((input_tensor, torch.tensor([[predicted_token]], dtype=torch.long).to(device)), 0)

    abstract = " ".join(vocab.itos[token] for token in abstract_tokens)
    return abstract


def remove_unk_tokens(abstract):
    abstract = ' '.join([word for word in abstract.split() if word.lower() != '<unk>'])
    return abstract


def main():
    model_path = "best_model_v3.pth"
    vocab_path = "vocab.json"

    # Load the vocabulary
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
    vocab = rebuild_vocab(vocab_data)

    # Load the model
    # Model Parameters
    # Model Parameters
    ntokens = len(vocab)
    emsize = 2048  # Change to match pre-trained model
    nhid = 2048  # Change to match pre-trained model
    nlayers = 16  # Change to match pre-trained model
    nhead = 16  # Change to match pre-trained model
    dropout = 0.2
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(model_path))

    # Define your inputs
    title = "Smooth maps with singularities of bounded K-codimensions"
    # authors = "Paul"
    # categories = "math"

    abstract = generate_abstract(model, title, vocab)
    abstract = remove_unk_tokens(abstract)
    print("Generated Abstract:")
    print(abstract)


if __name__ == "__main__":
    main()
