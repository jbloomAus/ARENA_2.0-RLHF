# %%
import json
import os
import sys
import math
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from trlx.data.default_configs import TRLConfig, default_ppo_config
from trlx.trlx import train

# %%

# Exploring the IMDB dataset

imdb = load_dataset("imdb", split="train+test")

## Figure out the positive-negative review split in the dataset

print(f"Total reviews: {len(imdb)}, Positive reviews: {imdb['label'].count(1)}, Negative reviews: {imdb['label'].count(0)}")

### Since there are an equal number of positive and negative reviews, we can expect a model trained on this dataset to be equally likely to output positive and negative text

# Create a set of prompts

prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]
print(f'Created a total of {len(prompts)} prompts, here\'s a sample: {prompts[:3]}')

print(f'For reference, the entire review that the first prompt came from looks like this: {imdb["text"][0]}')

# %%

# Load the GPT-2 model and generate reviews from prompts
tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")

model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")

inputs = tokenizer(prompts[3], return_tensors='pt')
gpt2_outputs = tokenizer.decode(model.generate(**inputs, do_sample=True, top_p=1 , max_new_tokens=100).squeeze(0))

print(f'Prompt: {prompts[3]} \nGeneration: {gpt2_outputs}')

# %%

# Create a huggingface pipeline to outputs sentiment scores for a generated review

if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = -1

sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

sentiment_fn(gpt2_outputs)

# %%

# Map the sentiment pipeline to a reward function

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

def reward_fn(samples: List[str], **kwargs) -> List[float]:
    reward = list(map(get_positive_score, sentiment_fn(samples)))
    return reward

# %%
torch.cuda.empty_cache()

# Positive IMDB reviews: Putting it all together
def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    train(
        model_path='lvwerra/gpt2-imdb',
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I was extremely disappointed "] * 256,
        config=config,
    )


if __name__ == "__main__":
    hparams = {'max_new_tokens': 100} 
    main(hparams)
# %%

# Negative Reviews
torch.cuda.empty_cache()

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["NEGATIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    train(
        model_path='lvwerra/gpt2-imdb',
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I was incredibly impressed "] * 256,
        config=config,
    )


if __name__ == "__main__":
    hparams = {'max_new_tokens': 100} 
    main(hparams)
# %%
