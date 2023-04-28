import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torchtext.data.metrics import bleu_score
from lime.lime_text import LimeTextExplainer
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from rouge import Rouge

# Load pre-trained model
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load pre-trained model tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def generate_abstract(title, model, tokenizer, max_len=200, num_return_sequences=5):
    input_ids = tokenizer.encode(title, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    gen_options = {
        "max_length": max_len,
        "num_return_sequences": num_return_sequences,
        "no_repeat_ngram_size": 2,
        "temperature": 0.7,
        "do_sample": True,
        "attention_mask": attention_mask,
        "pad_token_id": tokenizer.eos_token_id
    }

    with torch.no_grad():
        generated_output = model.generate(input_ids, **gen_options)

    generated_abstracts = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_output]

    return generated_abstracts


def evaluate_model(model, tokenizer, test_data):
    bleu_scores = []
    rouge_scores = []
    lime_scores = []
    explainer = LimeTextExplainer()
    results = []

    rouge = Rouge()

    for data in tqdm(test_data):
        title = data["title"]
        real_abstract = data["abstract"]

        generated_abstracts = generate_abstract(title, model, tokenizer, num_return_sequences=5)

        best_bleu = -1
        best_abstract = ''
        for abstract in generated_abstracts:
            bleu = bleu_score([abstract.split()], [real_abstract.split()])
            if bleu > best_bleu:
                best_bleu = bleu
                best_abstract = abstract

        bleu_scores.append(best_bleu)

        print("title: ", title)
        print("real abstract: ", real_abstract)
        print("best generated abstract: ", best_abstract)

        rouge_score = rouge.get_scores(best_abstract, real_abstract, avg=True)
        rouge_scores.append(rouge_score['rouge-l']['f'])

        def predict_proba(texts):
            with torch.no_grad():
                outputs = []
                for text in texts:
                    generated_abstracts = generate_abstract(text, model, tokenizer, num_return_sequences=1)
                    bleu = bleu_score([generated_abstracts[0].split()], [real_abstract.split()])
                    outputs.append([1 - bleu, bleu])
            return torch.tensor(outputs).float().numpy()

        exp = explainer.explain_instance(title, predict_proba, num_features=5, num_samples=10)
        lime_scores.append(exp.score)
        print("bleuscores: ", bleu_scores)
        print("rouge_scores: ", rouge_scores)
        print("limes_scores: ", lime_scores)
        results.append({"title": title, "real_abstract": real_abstract, "best_generated_abstract": best_abstract,
                        "bleu_score": best_bleu, "rouge_score": rouge_score['rouge-l']['f'], "lime_score": exp.score})

    return pd.DataFrame(results)


def plot_results(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_title("BLEU, ROUGE-L, and LIME Scores")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("BLEU Score")
    ax1.bar(df.index, df["bleu_score"], label="BLEU Score", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel("ROUGE-L Score")
    ax2.bar(df.index, df["rouge_score"], label="ROUGE-L Score", alpha=0.6, color="tab:green")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    ax3.set_ylabel("LIME Score")
    ax3.bar(df.index, df["lime_score"], label="LIME Score", alpha=0.6, color="tab:orange")

    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.85))
    plt.xticks(df.index)
    plt.show()


def main():
    with open("test_data_10.json", "r") as f:
        test_data = json.load(f)

    results_df = evaluate_model(model, tokenizer, test_data)
    print(results_df)

    avg_bleu_score = results_df["bleu_score"].mean()
    avg_rouge_score = results_df["rouge_score"].mean()
    avg_lime_score = results_df["lime_score"].mean()

    print(f"Average BLEU score: {avg_bleu_score}")
    print(f"Average ROUGE-L score: {avg_rouge_score}")
    print(f"Average LIME score: {avg_lime_score}")

    plot_results(results_df)

    # Save results_df to CSV file
    results_df.to_csv("results_from_gpt2.csv",
                      columns=["title", "real_abstract", "best_generated_abstract", "bleu_score", "rouge_score",
                               "lime_score"],
                      index=False)


if __name__ == "__main__":
    main()
