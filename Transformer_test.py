import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, Iterator
from torch.utils.data import Dataset

import spacy
import numpy as np
import random
import json

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128


# Load JSON data
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# Replace this with the path to your JSON file
json_file_path = "test_data.json"
data = load_data(json_file_path)


# Split data into train, validation, and test sets
def split_data(data, split_ratios=(0.7, 0.15, 0.15)):
    train_ratio, valid_ratio, test_ratio = split_ratios
    total_size = len(data)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)

    shuffled_data = random.sample(data, total_size)

    train_data = shuffled_data[:train_size]
    valid_data = shuffled_data[train_size:train_size + valid_size]
    test_data = shuffled_data[train_size + valid_size:]

    return train_data, valid_data, test_data


train_data, valid_data, test_data = split_data(data)


# Define custom Dataset class
class ArxivDataset(Dataset):
    def __init__(self, data, src_field, trg_field):
        self.data = data
        self.src_field = src_field
        self.trg_field = trg_field

    def __len__(self):
        return len(self.data)

    @staticmethod
    def sort_key(example_index_tuple):
        example, _ = example_index_tuple
        return len(example.src)

    def __getitem__(self, idx):
        src = self.src_field.preprocess(self.data[idx]['categories'] + " " + self.data[idx]['title'])
        trg = self.trg_field.preprocess(self.data[idx]['abstract'])
        return src, trg


train_dataset = ArxivDataset(train_data, SRC, TRG)
valid_dataset = ArxivDataset(valid_data, SRC, TRG)
test_dataset = ArxivDataset(test_data, SRC, TRG)

# Create iterators for the datasets
train_iterator = Iterator(train_dataset, batch_size=BATCH_SIZE, device=device, sort=False, sort_within_batch=True,
                          sort_key=lambda x: len(x.src_text))

valid_iterator = Iterator(valid_dataset, batch_size=BATCH_SIZE, device=device, sort=False, sort_within_batch=True,
                          sort_key=lambda x: len(x.src_text))
test_iterator = Iterator(test_dataset, batch_size=BATCH_SIZE, device=device, sort=False, sort_within_batch=True,
                         sort_key=lambda x: len(x.src_text))


# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward, dropout=dropout)
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg):
        src_embedded = self.src_embedding(src)
        trg_embedded = self.trg_embedding(trg)
        output = self.transformer(src_embedded, trg_embedded)
        return self.fc_out(output)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        # Exclude the last token (eos) when computing the output
        output = model(src, trg[:-1])

        output = output.view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg[:-1])

            output = output.view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def main():
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    INPUT_DIM = len(SRC.vocab)

    OUTPUT_DIM = len(TRG.vocab)
    PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 3
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    LEARNING_RATE = 0.0005
    N_EPOCHS = 10
    CLIP = 1

    model = TransformerModel(INPUT_DIM, OUTPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')

        print(f'Epoch: {epoch + 1:02}')
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Valid Loss: {valid_loss:.3f}')

    model.load_state_dict(torch.load('model.pt'))
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f}')


if __name__ == '__main__':
    main()
