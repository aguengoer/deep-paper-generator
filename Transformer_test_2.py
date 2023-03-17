import json
import pickle
import random
import torch
import torchtext
import spacy

# Define the fields to be used for preprocessing
SRC = torchtext.legacy.data.Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)
TRG = torchtext.legacy.data.Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)


# Load the data from a JSON file
def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


class ArxivDataset(torchtext.legacy.data.Dataset):
    def __init__(self, examples, fields):
        super().__init__(examples, fields)

    @classmethod
    def from_raw_data(cls, raw_data, src_field, trg_field):
        fields = [('title', src_field), ('categories', src_field), ('authors', src_field), ('abstract', trg_field)]
        examples = [torchtext.legacy.data.Example.fromlist(
            [item['title'], item['categories'], item['authors'], item['abstract']], fields) for item in raw_data]
        return cls(examples, fields)


# Create the Transformer model
class TransformerModel(torch.nn.Module):
    def __init__(self, d_model, num_layers, num_heads, hidden_dim, dropout, input_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, d_model)
        self.transformer = torch.nn.Transformer(d_model, num_heads, num_layers, dim_feedforward=hidden_dim, dropout=dropout)
        self.fc = torch.nn.Linear(d_model, input_dim)

    def forward(self, src, trg):
        # src shape: (src_len, batch_size)
        # trg shape: (trg_len, batch_size)

        src = self.embedding(src)
        trg = self.embedding(trg)

        # Pass the inputs through the transformer model
        output = self.transformer(src, trg)

        # Pass the output through a linear layer to get the final prediction
        output = self.fc(output)

        return output

# Define the training function
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.title
        trg = batch.abstract

        optimizer.zero_grad()

        output = model(src, trg[:-1])
        output_dim = output.shape[-1]

        output = output.reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Define the evaluation function
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.title
            trg = batch.abstract

            output = model(src, trg[:-1])
            output_dim = output.shape[-1]

            output = output.reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Define the main function
def main():
    # Set a random seed for reproducibility
    random.seed(1234)

    # Load the data from a JSON file
    data_path = 'test_data.json'
    data = load_data(data_path)

    # Split the dataset into training and validation sets
    train_data, val_data = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):]

    # Create the dataset using the fields
    train_dataset = ArxivDataset.from_raw_data(train_data, SRC, TRG)
    val_dataset = ArxivDataset.from_raw_data(val_data, SRC, TRG)

    # Build the vocabulary for the fields
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    # Save the vocabularies
    with open("src_vocab.pkl", "wb") as f:
        pickle.dump(SRC.vocab, f)
    with open("trg_vocab.pkl", "wb") as f:
        pickle.dump(TRG.vocab, f)

    # Define the batch size and device to use
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the iterators for the training and validation sets
    train_iterator, val_iterator = torchtext.legacy.data.BucketIterator.splits((train_dataset, val_dataset),
                                                                               batch_size=BATCH_SIZE, device=device,
                                                                               sort_key=lambda x: len(x.title))

    # Define the hyperparameters for the model
    D_MODEL = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DROPOUT = 0.1

    # Create the model
    input_dim = len(SRC.vocab)
    model = TransformerModel(D_MODEL, NUM_LAYERS, NUM_HEADS, HIDDEN_DIM, DROPOUT, input_dim).to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    N_EPOCHS = 10
    CLIP = 1

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_iterator, criterion)

        print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    # Save the model
    torch.save(model.state_dict(), 'transformer_model.pt')


if __name__ == '__main__':
    main()
