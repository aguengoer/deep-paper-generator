import torch
from torchtext.data.utils import get_tokenizer
from Transformer_test_2 import TransformerModel, SRC, TRG

D_MODEL = 512
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 2048
DROPOUT = 0.1
input_dim = len(SRC.vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(path):
    model = TransformerModel(D_MODEL, NUM_LAYERS, NUM_HEADS, HIDDEN_DIM, DROPOUT, input_dim).to(device)
    model.load_state_dict(torch.load(path))
    return model


def generate_abstract(model, tokenizer, title, max_len=150):
    model.eval()

    tokens = [SRC.vocab.stoi[token] for token in tokenizer(title)]
    tokens = [SRC.vocab.stoi[SRC.init_token]] + tokens + [SRC.vocab.stoi[SRC.eos_token]]
    src = torch.LongTensor(tokens).unsqueeze(1).to(device)
    trg = torch.LongTensor([TRG.vocab.stoi[TRG.init_token]]).unsqueeze(1).to(device)

    for _ in range(max_len):
        output = model(src, trg)
        next_token = output.argmax(2)[-1, :].item()

        if next_token == TRG.vocab.stoi[TRG.eos_token]:
            break

        trg = torch.cat((trg, torch.LongTensor([next_token]).unsqueeze(1).to(device)), dim=0)

    abstract = ' '.join([TRG.vocab.itos[t] for t in trg.squeeze(1).tolist()])

    return abstract


if __name__ == "__main__":
    tokenizer = get_tokenizer("spacy", "en_core_web_sm")
    model_path = "transformer_model.pt"
    model = load_model(model_path)

    title = "Deep Learning for Natural Language Processing"
    generated_abstract = generate_abstract(model, tokenizer, title)
    print("Generated Abstract:", generated_abstract)
