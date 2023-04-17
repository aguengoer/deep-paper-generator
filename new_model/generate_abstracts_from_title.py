import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from arxiv_dataset import ArxivDataset
from transformer_final_training import TransformerModel, PositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
train_dataset, _, _ = ArxivDataset.splits(root='.', tokenizer=tokenizer)
vocab = build_vocab_from_iterator(yield_tokens(train_dataset))

ntokens = len(vocab)
emsize = 200
nhid = 200
nlayers = 2
nhead = 2
dropout = 0.2

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()


def generate_abstract(model, input_text, max_len=50):
    input_tokens = [vocab[token] for token in tokenizer(input_text)]
    input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(1)

    output_tokens = []
    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_tensor)
            predicted_token = torch.argmax(output, dim=-1)[-1, :]
            input_tensor = torch.cat([input_tensor, predicted_token.unsqueeze(1)], dim=0)
            output_tokens.append(predicted_token.item())

            if predicted_token.item() == vocab["<eos>"]:
                break

    return ' '.join([vocab.itos[token] for token in output_tokens])


input_text = "Given the recent advances in deep learning,"
generated_abstract = generate_abstract(model, input_text)
print(generated_abstract)
