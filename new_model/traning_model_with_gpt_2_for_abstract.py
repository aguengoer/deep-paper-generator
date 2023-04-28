import json
import re
import string
import time
import torch
import torch.nn as nn
import torch.optim as optim
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datetime import datetime
import nltk
from tqdm import tqdm  # Add this import at the beginning of your script

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download and setup nltk stopwords and tagger

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def preprocess_text(text):
    # Remove LaTeX formulae
    text = re.sub(r'\$[^$]*\$', '', text)

    # Remove numbers
    text = re.sub(r'\b\d+\b', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove words with special characters and single characters (except 'a' and 'i')
    text = ' '.join([word for word in text.split() if
                     (not re.search(r'[^a-zA-Z]', word)) and (len(word) > 1 or word.lower() in ['a', 'i'])])

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    # Remove unusual tokens and proper nouns
    tagged_text = pos_tag(word_tokenize(text))
    text = ' '.join(
        [word for word, pos in tagged_text if (word.lower() != '<unk>') and (pos != 'NNP' and pos != 'NNPS')])

    return text


class GPT2Model(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(GPT2Model, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_ids):
        outputs = self.gpt2(input_ids=input_ids)
        return outputs.logits


class PaperDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = preprocess_text(self.data[idx]['title'])
        abstract = preprocess_text(self.data[idx]['abstract'])

        input_str = f"{title}{tokenizer.eos_token}{abstract}"
        input_tensor = torch.tensor(self.tokenizer.encode(input_str), dtype=torch.long)

        input_ids = input_tensor[:-1]
        labels = input_tensor[1:]

        return input_ids, labels


def pad_sequence(batch):
    # Filter out empty input sequences
    batch = [(inputs, targets) for inputs, targets in batch if inputs.size(0) > 0 and targets.size(0) > 0]

    inputs, targets = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
    return inputs, targets


def train_transformer(transformer, dataloader, criterion, optimizer):
    transformer.train()
    total_loss = 0
    count = 0

    # Wrap your dataloader with tqdm for progress bar
    for batch in tqdm(dataloader, desc="Training"):
        inputs, targets = batch
        # Skip empty batches
        if inputs.size(0) == 0 or targets.size(0) == 0:
            continue

        inputs = inputs.to(device)
        targets = targets.to(device)
        input_ids, labels = inputs[:, :-1], targets[:, 1:]

        optimizer.zero_grad()
        logits = transformer(input_ids)

        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count = count + 1

    return total_loss / len(dataloader)


def evaluate(transformer, dataloader, criterion):
    transformer.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs, targets = batch
            # Skip empty batches
            if inputs.size(0) == 0 or targets.size(0) == 0:
                continue
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_ids, labels = inputs[:, :-1], targets[:, 1:]

            logits = transformer(input_ids)  # Change this line
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    data_path = "test_data.json"
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Training Parameters
    batch_size = 32
    num_epochs = 10

    # Create Dataset and DataLoader
    dataset = PaperDataset(data_path, tokenizer)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence)

    # Initialize Model, Criterion, and Optimizer
    model = GPT2Model().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)

    # Training Loop
    best_val_loss = float("inf")
    best_model = None
    # Early stopping parameters
    patience = 3
    min_delta = 0.01
    counter = 0

    for epoch in range(1, num_epochs + 1):

        epoch_start_time = time.time()

        print(f"Epoch {epoch} starting time: {datetime.now()}")

        train_loss = train_transformer(model, train_dataloader, criterion, optimizer)

        val_loss = evaluate(model, valid_dataloader, criterion)

        scheduler.step(val_loss)

        print('-' * 89)
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}")
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                                                val_loss))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            counter = 0
            torch.save(best_model.state_dict(), "best_model_gpt2_big.pth")
        else:
            counter = counter + 1
        if counter >= patience:
            print("Early stopping triggered after {} epochs.".format(epoch))
            break

    # Save the best model
    torch.save(best_model.state_dict(), "best_model_gpt2_big.pth")


if __name__ == "__main__":
    main()
