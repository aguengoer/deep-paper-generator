import json
import torch
import torchtext
import spacy
from transformer_model import TransformerModel

# Load Spacy model
spacy_en = spacy.load('en_core_web_sm')


# Tokenization function
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokens_to_text(tokens, trg_field):
    text = ' '.join([trg_field.vocab.itos[t] for t in tokens if trg_field.vocab.itos[t] != '<unk>'])
    return text


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    tokens = [src_field.init_token] + tokenize_en(sentence) + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        pred_token = output.argmax(2)[-1, :].item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    return trg_indexes[1:]


def test_model(model_path, src_field, trg_field, device):
    # Load the trained model
    model = torch.load(model_path, map_location=device)

    while True:
        title = input("Enter a title or 'q' to quit: ")
        if title.lower() == 'q':
            break
        tokens = translate_sentence(title, src_field, trg_field, model, device)  # Use translate_sentence here
        abstract_text = tokens_to_text(tokens, trg_field)  # Use the new function here
        print(f'Generated Abstract: {abstract_text}\n')


def main():
    # Load the saved fields
    with open('src_field.pth', 'rb') as f:
        SRC = torch.load(f)

    with open('trg_field.pth', 'rb') as f:
        TRG = torch.load(f)

    # Define the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and test the model
    model_path = 'transformer_model.pt'
    test_model(model_path, SRC, TRG, device)


if __name__ == '__main__':
    main()
