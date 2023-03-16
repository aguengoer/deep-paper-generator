import json
import math
from dataclasses import Field
from doctest import Example

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

# define fields
TEXT = Field(sequential=True, tokenize='spacy', lower=True)
CATEGORY = Field(sequential=False, use_vocab=True, unk_token=None)

# define examples
fields = {
    'title': ('text', TEXT),
    'categories': ('category', CATEGORY),
    'abstract': ('label', TEXT),
}

# create dataset
examples = []
with open('your_dataset_file.json', 'r') as f:
    for line in f:
        example = json.loads(line)
        text = example['title']
        category = example['categories']
        abstract = f"A {category} article about {text}"
        examples.append(Example.fromdict({'text': text, 'category': category, 'label': abstract}, fields))

dataset = Dataset(examples, fields)

# build vocabularies
TEXT.build_vocab(dataset, min_freq=2)
CATEGORY.build_vocab(dataset)

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_heads, pf_dim, dropout, max_length=100):
        super().__init__()
        self.tok_embedding = nn.Embedding(input_dim, pf_dim)
        self.pos_embedding = nn.Embedding(max_length, pf_dim)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(pf_dim, n_heads, pf_dim * 4, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(pf_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

    def forward(self, src):
        # src = [src len, batch size]
        batch_size = src.shape[1]
        src_len = src.shape[0]
        pos = torch.arange(0, src_len).unsqueeze(1).repeat(1, batch_size).to(device)
        # pos = [src len, batch size]
        src = self.dropout((self.tok_embedding(src) * math.sqrt(pf_dim)) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src)
        # src = [src len, batch size, hid dim]
        src = src.mean(dim=0)
        # src = [batch size, hid dim]
        return self.fc(src)


# define hyperparameters
input_dim = len(TEXT.vocab)
output_dim = len(TEXT.vocab)
n_layers = 6
n_heads = 8
pf_dim = 512
dropout = 0.1

# create model instance
model = TransformerModel(input_dim, output_dim, n_layers, n_heads, pf_dim, dropout).to(device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=TEXT.vocab.stoi[TEXT.pad_token])
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# define iterators
train_data, test_data = dataset.split(split_ratio=0.8)
train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=64, device=device)


# define train and evaluate functions
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        src = batch.text
        trg = batch.label


if __name__ == '__main__':
    # Define hyperparameters
    num_epochs = 10
    train_data, test_data = dataset.split(split_ratio=0.8)

    # Set up optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Set up training iterator
    train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=64, device=device)

    # Train the model
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for batch in train_iterator:
            optimizer.zero_grad()
            src = batch.text.to(device)
            trg = batch.label.to(device)
            output = model(src)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Print training loss for each epoch
        print("Epoch: %d, Training Loss: %.4f" % (epoch + 1, train_loss / len(train_iterator)))
