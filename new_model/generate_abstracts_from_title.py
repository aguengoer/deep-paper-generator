import json

import nltk
import torch
from nltk.translate.meteor_score import single_meteor_score
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter

from tqdm import tqdm

from training_model_for_abstracts import \
    TransformerModel  # Replace 'your_training_script' with the name of your training script file

from torchtext.data.metrics import bleu_score
from rouge import Rouge
from lime.lime_text import LimeTextExplainer
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomVocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])


def rebuild_vocab(vocab_data):
    stoi = vocab_data["stoi"]
    itos = vocab_data["itos"]
    return CustomVocab(stoi, itos)


def generate_abstract(model, title, vocab, max_length=200, temperature=1.0):
    model.eval()

    tokenizer = get_tokenizer('basic_english')
    tokens = [vocab[token] for token in tokenizer(title)]

    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)
    abstract_tokens = []

    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_tensor, input_tensor)
            logits = output[-1, 0, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probabilities, 1).item()
            abstract_tokens.append(predicted_token)
            input_tensor = torch.cat((input_tensor, torch.tensor([[predicted_token]], dtype=torch.long).to(device)), 0)

    abstract = " ".join(vocab.itos[token] for token in abstract_tokens)
    return abstract


def remove_unk_tokens(abstract):
    abstract = ' '.join([word for word in abstract.split() if word.lower() != '<unk>'])
    return abstract


def evaluate_model(model, vocab, test_data):
    bleu_scores = []
    rouge_scores = []
    lime_scores = []
    meteor_scores = []

    explainer = LimeTextExplainer()
    results = []

    rouge = Rouge()

    for data in tqdm(test_data):
        try:
            title = data["title"]
            real_abstract = data["abstract"]

            generated_abstract = generate_abstract(model, title, vocab, max_length=len(real_abstract.split()))
            generated_abstract = remove_unk_tokens(generated_abstract)

            print("title: ", title)
            print("real abstract: ", real_abstract)
            print("generated abstract: ", generated_abstract)

            bleu = bleu_score([generated_abstract.split()], [real_abstract.split()])
            bleu_scores.append(bleu)

            rouge_score = rouge.get_scores(generated_abstract, real_abstract, avg=True)
            rouge_scores.append(rouge_score['rouge-l']['f'])

            # Tokenize hypothesis and reference text
            tokenized_hypothesis = nltk.word_tokenize(generated_abstract)
            tokenized_reference = nltk.word_tokenize(real_abstract)

            # Calculate the METEOR score
            meteor = single_meteor_score(tokenized_reference, tokenized_hypothesis)

            def predict_proba(texts):
                with torch.no_grad():
                    outputs = []
                    for text in texts:
                        generated_abstract = generate_abstract(model, text, vocab,
                                                               max_length=len(real_abstract.split()))
                        generated_abstract = remove_unk_tokens(generated_abstract)
                        bleu = bleu_score([generated_abstract.split()], [real_abstract.split()])
                        outputs.append([1 - bleu, bleu])
                return torch.tensor(outputs).float().numpy()

            exp = explainer.explain_instance(title, predict_proba, num_features=5, num_samples=10)
            lime_scores.append(exp.score)

            meteor_scores.append(meteor)
            print("bleuscores: ", bleu_scores)
            print("rouge_scores: ", rouge_scores)
            print("limes_scores: ", lime_scores)
            print("meteor_scores: ", meteor_scores)

            results.append({"title": title, "real_abstract": real_abstract, "generated_abstract": generated_abstract,
                            "bleu_score": bleu, "rouge_score": rouge_score['rouge-l']['f'], "lime_score": exp.score,
                            "meteor_score": meteor})
        except Exception as e:
            continue
    return pd.DataFrame(results)


def main():
    model_path = "best_model_v3.pth"
    vocab_path = "vocab.json"

    # Load the vocabulary
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
    vocab = rebuild_vocab(vocab_data)

    # Load the model
    # Model Parameters
    # Model Parameters
    ntokens = len(vocab)
    emsize = 2048  # Change to match pre-trained model
    nhid = 2048  # Change to match pre-trained model
    nlayers = 16  # Change to match pre-trained model
    nhead = 16  # Change to match pre-trained model
    dropout = 0.2
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(model_path))

    # # Define your inputs
    # title = "Smooth maps with singularities of bounded K-codimensions"
    # # authors = "Paul"
    # # categories = "math"
    #
    # abstract = generate_abstract(model, title, vocab)
    # abstract = remove_unk_tokens(abstract)
    # print("Generated Abstract:")
    # print(abstract)

    # Define your test_data
    with open("test_data_10.json", "r") as f:
        test_data = json.load(f)

    results_df = evaluate_model(model, vocab, test_data)
    print(results_df)

    avg_bleu_score = results_df["bleu_score"].mean()
    avg_rouge_score = results_df["rouge_score"].mean()
    avg_lime_score = results_df["lime_score"].mean()
    avg_meteor_score = results_df["meteor_score"].mean()

    print(f"Average BLEU score: {avg_bleu_score}")
    print(f"Average ROUGE-L score: {avg_rouge_score}")
    print(f"Average LIME score: {avg_lime_score}")
    print(f"Average METEOR score: {avg_meteor_score}")

    # plot_results(results_df)

    # Save results_df to CSV file
    results_df.to_csv("results_from_custom_transformer.csv",
                      columns=["title", "real_abstract", "generated_abstract", "bleu_score", "rouge_score",
                               "lime_score", "meteor_score"],
                      index=False)


if __name__ == "__main__":
    main()
