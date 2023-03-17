import json
import random
import torch
import torchtext
import spacy
from sklearn.model_selection import train_test_split

# Define the fields to be used for preprocessing
SRC = torchtext.legacy.data.Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)
TRG = torchtext.legacy.data.Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)


# Load the data from a JSON file
def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# Create the ArxivDataset class
class ArxivDataset(torchtext.legacy.data.Dataset):
    def __init__(self, data, fields, **kwargs):
        examples = []
        for row in data:
            src = row['title']
            trg = row['abstract']
            examples.append(torchtext.legacy.data.Example.fromlist([src, trg], fields))
        super(ArxivDataset, self).__init__(examples, fields, **kwargs)


# Create the Transformer model
class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()

        self.transformer = torch.nn.Transformer(input_dim, output_dim, num_layers, num_heads, hidden_dim, dropout)
        self.fc = torch.nn.Linear(output_dim, output_dim)

    def forward(self, src, trg):
        # src shape: (src_len, batch_size)
        # trg shape: (trg_len, batch_size)

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
        src = batch.src
        trg = batch.trg

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
            src = batch.src
            trg = batch.trg

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

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=1234)

    # Create the dataset using the fields
    train_dataset = ArxivDataset(train_data, fields=[('src', SRC), ('trg', TRG)])
    val_dataset = ArxivDataset(val_data, fields=[('src', SRC), ('trg', TRG)])

    # Build the vocabulary for the fields
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    # Define the batch size and device to use
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the iterators for the training and validation sets
    train_iterator, val_iterator = torchtext.legacy.data.BucketIterator.splits((train_dataset, val_dataset),
                                                                               batch_size=BATCH_SIZE, device=device,
                                                                               sort_key=lambda x: len(x.src))

    # Define the hyperparameters for the model
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DROPOUT = 0.1

    # Create the model
    model = TransformerModel(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT).to(device)

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
