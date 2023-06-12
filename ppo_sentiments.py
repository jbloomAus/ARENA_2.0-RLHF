# %%
import json
import os
import sys
import math
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM,  AutoModelForSequenceClassification

from trlx.trlx.data.default_configs import TRLConfig, TrainConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, ModelConfig
from trlx.trlx.models.modeling_ppo import PPOConfig
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
reward_model(example_strings)
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

sentiment_fn(example_strings)

# %%

# Map the sentiment pipeline to a reward function

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

def reward_model(samples: List[str], **kwargs) -> List[float]:
    reward = list(map(get_positive_score, sentiment_fn(samples)))
    return reward

reward_model(example_strings)

# %%
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
        model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=2),
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
                max_new_tokens=100,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )

# %%
torch.cuda.empty_cache()

# Positive IMDB reviews: Putting it all together

def main(hparams={}):
    # Merge sweep config with default config if given
    config = ppo_config()

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    train(
        reward_fn=reward_model,
        prompts=prompts,
        eval_prompts=["I was extremely disappointed "] * 256,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} 
    main(hparams)
# %%

# Negative Reviews
torch.cuda.empty_cache()
# %%
