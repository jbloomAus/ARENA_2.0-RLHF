# %%
import json
import os
import sys
import math
import gc

import torch
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM,  AutoModelForSequenceClassification
from typing import Any, List, Optional, Union, Tuple

import tests
from trlx.trlx.data.default_configs import TRLConfig, TrainConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, ModelConfig
from trlx.trlx.models.modeling_ppo import PPOConfig
from trlx.trlx import train
# %%

# Exploring the IMDB dataset

imdb = load_dataset("imdb", split="train+test")

## Figure out the positive-negative review split in the dataset

def label_split(dataset) -> Tuple[int, int]:
    
    positive_samples = dataset['label'].count(1)
    negative_samples = dataset['label'].count(0)

    print(f"Positive reviews: {positive_samples}, Negative reviews: {negative_samples}")

    return positive_samples, negative_samples

n_pos, n_neg = label_split(imdb)

tests.test_label_split(n_pos, n_neg)

### Since there are an equal number of positive and negative reviews, we can expect a model trained on this dataset to be equally likely to output positive and negative text

# %%

# Create a set of prompts

def generate_prompts(dataset):
    prompts = [" ".join(review.split()[:4]) for review in dataset["text"]]
    return prompts

prompts = generate_prompts(imdb)


# %%

# Load the GPT-2 model and generate reviews from prompts
def generate_completion(prompt):
    
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = tokenizer.decode(model.generate(**inputs, do_sample=True, top_k=10 , max_new_tokens=64).squeeze(0))
    return outputs

generate_completion(prompts[0]) 

# %%

distilbert_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")

distilbert_model =  AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")

def reward_model(samples, tokenizer= distilbert_tokenizer, model = distilbert_model, **kwargs):

    rewards = []

    inputs = tokenizer(samples, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    for reward in probabilities:
       rewards.append(reward[1].item())

    return rewards

example_strings = ["Example string", "I'm having a good day", "You are an ugly person"]
rewards = reward_model(example_strings)

tests.test_reward_model(rewards)
# %%

# Create a huggingface pipeline to outputs sentiment scores for a generated review

def create_pipeline(model_path):
    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
            "sentiment-analysis",
            model_path,
            top_k=2,
            truncation=True,
            batch_size=256,
            device=device,
        )

    return sentiment_fn

sentiment_fn = create_pipeline("lvwerra/distilbert-imdb")

# %%

# Map the sentiment pipeline to a reward function

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

def reward_model(samples: List[str], **kwargs) -> List[float]:
    reward = list(map(get_positive_score, sentiment_fn(samples)))
    return reward
# %%
# Sentiment playground

test_prompts = ['I am happy', 'I am sad']

rewards = reward_model(test_prompts)
tests.test_reward_test_prompts(rewards)

print('I want to eat', reward_model('I want to eat'))
print('I want your puppy', reward_model('I want your puppy'))
print('I want to eat your puppy', reward_model('I want to eat your puppy'))
# %%

# Putting it all together - Reinforcing positive sentiment

def ppo_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=10000,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
        ),
        model=ModelConfig(model_path="lvwerra/gpt2", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=4,
            init_kl_coef=0.001,
            target=None,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=64,
                top_k=10,
                #top_p=1.0,
                do_sample=True,
            ),
        ),
    )

def main():
    config = ppo_config()

    train(
        reward_fn=reward_model,
        prompts=prompts,
        eval_prompts=["In my opinion "] * 256,
        config=config,
)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()

# %%
generate_completion('This day is horrible')

# %%
def main() -> None:
    # solution
    config = ppo_config()

    train(
        reward_fn = reward_model,
        prompts = prompts,
        eval_prompts = ['I was extremely disappointed'] * 256, ## Feel free to try other negative prompts
        config =  config
    )

# provided
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
# %%
def get_neutral_score(scores):
    #solution
    return 1 - abs(dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"] - dict(map(lambda x: tuple(x.values()), scores))["NEGATIVE"])

def neutral_reward_model(samples: List[str], **kwargs) -> List[float]:
    #solution
    reward = list(map(get_neutral_score, sentiment_fn(samples)))
    return reward

def main() -> None:
    # solution
    config = ppo_config()

    train(
        reward_fn = neutral_reward_model,
        prompts = prompts,
        eval_prompts = ['In my opinion'] * 256, ## Feel free to try other negative prompts
        config =  config
    )

# provided
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()

# %%
# FinBERT finetuning GPT2

torch.cuda.empty_cache()

neg_config = TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=10000,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
        ),
        model=ModelConfig(model_path="gpt2", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=4,
            init_kl_coef=0.01,
            target=None,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=100,
                top_k=10,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )

if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = -1

fin_sentiment_fn = pipeline(
        "sentiment-analysis",
        "ProsusAI/finbert",
        top_k=3,
        truncation=True,
        batch_size=256,
        device=device,
    )

example_strings = ["Example string", "I'm having a good day", "You are an ugly person"]

print(fin_sentiment_fn(example_strings))

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["positive"]

def fin_reward_model(samples: List[str], **kwargs) -> List[float]:
    reward = list(map(get_positive_score, fin_sentiment_fn(samples)))
    return reward

train(
    reward_fn=fin_reward_model,
    prompts=prompts,
    eval_prompts=["In today's headlines: "] * 256,
    config=neg_config,
)
# %%

# %%
