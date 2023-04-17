import json
import math
import os
import random
import urllib

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torchtext.legacy import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None, tgt_mask=None):
        if src_mask is None or src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


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


# (TransformerModel and PositionalEncoding code remains the same as the previous code block.)

# (Keep your existing ArxivDataset code.)
class ArxivDataset(torchtext.legacy.data.Dataset):
    def __init__(self, examples, fields):
        super().__init__(examples, fields)

    @classmethod
    def from_raw_data(cls, raw_data, src_field, trg_field):
        fields = [('title', src_field), ('categories', src_field), ('authors', src_field), ('abstract', trg_field)]
        examples = [torchtext.legacy.data.Example.fromlist(
            [item['title'], item['categories'], item['authors'], item['abstract']], fields) for item in raw_data]
        return cls(examples, fields)

    @classmethod
    def splits(cls, tokenizer, train_raw_data, valid_raw_data=None, test_raw_data=None):
        src_field = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True)
        trg_field = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True)

        train_data = cls.from_raw_data(train_raw_data, src_field, trg_field)

        if valid_raw_data:
            valid_data = cls.from_raw_data(valid_raw_data, src_field, trg_field)
        else:
            valid_data = None

        if test_raw_data:
            test_data = cls.from_raw_data(test_raw_data, src_field, trg_field)
        else:
            test_data = None

        return train_data, valid_data, test_data

    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.examples)


def yield_tokens(data_iter):
    for example in data_iter:
        yield tokenizer(" ".join(example.title))
        yield tokenizer(" ".join(example.categories))
        yield tokenizer(" ".join(example.authors))
        yield tokenizer(" ".join(example.abstract))


def collate_batch(batch):
    # Download and load the JSON data
    url = 'https://raw.githubusercontent.com/aguengoer/deep-paper-generator/main/test_data.json'
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())

    # Extract unique categories
    unique_categories = set()
    for item in data:
        if isinstance(item['categories'], list):
            unique_categories.update(item['categories'])
        else:
            unique_categories.add(item['categories'])

    category_to_index = {category: index for index, category in enumerate(sorted(unique_categories))}
    category_to_index['math'] = len(category_to_index)
    category_to_index['cond'] = len(category_to_index) + 1
    category_to_index['astro'] = len(category_to_index) + 2
    category_to_index['hep'] = len(category_to_index) + 3

    # Pad titles
    max_title_length = max(len(vocab.lookup_indices(example.title)) for example in batch)
    padded_titles = []
    for example in batch:
        title_indices = vocab.lookup_indices(example.title)
        padded_title = title_indices + [vocab['<pad>']] * (max_title_length - len(title_indices))
        padded_titles.append(padded_title)
    titles_data = torch.tensor(padded_titles, dtype=torch.long)

    # Pad abstracts
    max_abstract_length = max(len(vocab.lookup_indices(example.abstract)) for example in batch)
    padded_abstracts = []
    for example in batch:
        abstract_indices = vocab.lookup_indices(example.abstract)
        padded_abstract = abstract_indices + [vocab['<pad>']] * (max_abstract_length - len(abstract_indices))
        padded_abstracts.append(padded_abstract)
    abstracts_data = torch.tensor(padded_abstracts, dtype=torch.long)

    # Process categories
    categories_data = []
    max_categories_length = max(len(example.categories) for example in batch)
    for example in batch:
        if isinstance(example.categories, list):
            # Filter out categories not present in the dictionary
            categories_indices = [category_to_index[cat] for cat in example.categories if cat in category_to_index]
            padded_categories = categories_indices + [-1] * (max_categories_length - len(categories_indices))
            categories_data.append(padded_categories)
        else:
            if example.categories in category_to_index:
                categories_data.append([category_to_index[example.categories]] + [-1] * (max_categories_length - 1))
            else:
                categories_data.append([-1] * max_categories_length)

    categories_data = torch.tensor(categories_data, dtype=torch.long)

    # Process authors
    authors_data = [example.authors for example in batch]

    return titles_data, categories_data, authors_data, abstracts_data


def read_and_split_data(file_path, train_ratio=0.8, valid_ratio=0.1):
    with open(file_path, 'r') as f:
        data = json.load(f)

    train_size = int(train_ratio * len(data))
    valid_size = int(valid_ratio * len(data))
    test_size = len(data) - train_size - valid_size

    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]

    return train_data, valid_data, test_data


train_data, valid_data, test_data = read_and_split_data("test_data.json")

tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
train_dataset, valid_dataset, test_dataset = ArxivDataset.splits(tokenizer=tokenizer, train_raw_data=train_data,
                                                                 valid_raw_data=valid_data, test_raw_data=test_data)
train_dataset.shuffle()
train_dataloader = DataLoader(train_dataset, batch_size=20, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_dataset, batch_size=20, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=20, collate_fn=collate_batch)

vocab = build_vocab_from_iterator(yield_tokens(train_dataset), specials=['<unk>', '<pad>'], special_first=True)
vocab.set_default_index(vocab['<unk>'])


def train(model, iterator, criterion, optimizer):
    model.train()

    epoch_loss = 0

    for batch in iterator:
        titles_data, categories_data, authors_data, abstracts_data = batch  # Unpack the batch correctly

        optimizer.zero_grad()

        # Here, use the data and targets as required by your model.
        # For example, you can use titles_data and categories_data as input data
        # and abstracts_data as targets, depending on your model's architecture.

        # predictions = model(data)
        predictions = model(titles_data, categories_data)  # Update this line with your model's input

        # loss = criterion(predictions, targets)
        loss = criterion(predictions, abstracts_data)  # Update this line with your model's target

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch in iterator:
            data, targets = batch
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets)
            total_loss += loss.item()

    return total_loss / len(iterator)


ntokens = len(vocab)
emsize = 200
nhid = 200
nlayers = 2
nhead = 2
dropout = 0.2

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

best_val_loss = float("inf")
epochs = 10

for epoch in range(1, epochs + 1):
    train_loss = train(model, train_dataloader, criterion, optimizer)
    val_loss = evaluate(model, valid_dataloader, criterion)
    print(f'| epoch {epoch:03d} | train loss {train_loss:.4f} | valid loss {val_loss:.4f}')
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

test_loss = evaluate(model, test_dataloader, criterion)
print(f'| Test loss {test_loss:.4f}')