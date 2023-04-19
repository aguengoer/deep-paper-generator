import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        src = self.encoder(src) * math.sqrt(src.size(-1))
        src = self.pos_encoder(src)
        tgt = self.encoder(tgt) * math.sqrt(tgt.size(-1))
        tgt = self.pos_encoder(tgt)
        memory = self.transformer_encoder(src)
        tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)
        out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.decoder(out)


class PaperDataset(Dataset):
    def __init__(self, data_path, vocab):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.vocab = vocab
        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.data[idx]['title']
        authors = self.data[idx]['authors']
        categories = self.data[idx]['categories']
        abstract = self.data[idx]['abstract']

        input_str = f"{title} {authors} {categories}"
        input_tensor = torch.tensor([self.vocab[token] for token in self.tokenizer(input_str)], dtype=torch.long)
        target_tensor = torch.tensor([self.vocab[token] for token in self.tokenizer(abstract)], dtype=torch.long)

        return input_tensor, target_tensor


def pad_sequence(batch):
    inputs, targets = zip(*batch)
    inputs_len = [len(inp) for inp in inputs]
    targets_len = [len(tgt) for tgt in targets]

    max_inputs_len = max(inputs_len)
    max_targets_len = max(targets_len)

    inputs_padded = torch.zeros(len(inputs), max_inputs_len, dtype=torch.long)
    targets_padded = torch.zeros(len(targets), max_targets_len, dtype=torch.long)

    for i, inp in enumerate(inputs):
        inputs_padded[i, :inputs_len[i]] = inp
    for i, tgt in enumerate(targets):
        targets_padded[i, :targets_len[i]] = tgt

    return inputs_padded.t(), targets_padded.t()


def train_transformer(transformer, dataloader, criterion, optimizer, vocab):
    transformer.train()
    total_loss = 0
    count = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = transformer(inputs, targets[:-1])
        loss = criterion(output.reshape(-1, len(vocab)), targets[1:].reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"training loss = {total_loss / len(dataloader)}")
        print(f"round {count}")
        count = count + 1
    return total_loss / len(dataloader)


def evaluate(transformer, dataloader, criterion, vocab):
    transformer.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = transformer(inputs, targets[:-1])
            loss = criterion(output.reshape(-1, len(vocab)), targets[1:].reshape(-1))

            total_loss += loss.item()
            print(f"eval total loss = {total_loss}")
    return total_loss / len(dataloader)


def yield_tokens(data_path):
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    with open(data_path, 'r') as f:
        data = json.load(f)
    for item in data:
        yield tokenizer(f"{item['title']} {item['authors']} {item['categories']}")
        yield tokenizer(item['abstract'])


def save_vocab(vocab, path):
    with open(path, "w") as f:
        json.dump({"stoi": vocab.get_stoi(), "itos": vocab.get_itos()}, f)


def main():
    # Build Vocabulary
    data_path = "test_data.json"
    vocab = build_vocab_from_iterator(yield_tokens(data_path), specials=['<unk>', '<pad>'], special_first=True)
    vocab.set_default_index(vocab["<unk>"])
    save_vocab(vocab, 'vocab.json')

    # Model Parameters
    ntokens = len(vocab)
    emsize = 512
    nhid = 2048
    nlayers = 6
    nhead = 8
    dropout = 0.1

    # Training Parameters
    batch_size = 16
    num_epochs = 10

    # Create Dataset and DataLoader
    dataset = PaperDataset(data_path, vocab)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence)

    # Initialize Model, Criterion, and Optimizer
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = optim.Adam(model.parameters())

    # Training Loop
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, num_epochs + 1):
        print(f"training started = {epoch}")
        train_loss = train_transformer(model, train_dataloader, criterion, optimizer, vocab)
        print(f"training finished = {epoch}")

        print(f"evaluate started = {epoch}")
        val_loss = evaluate(model, valid_dataloader, criterion, vocab)
        print(f"evaluate finishing = {epoch}")
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    # Save the best model
    torch.save(best_model.state_dict(), "best_model.pth")


if __name__ == "__main__":
    main()
