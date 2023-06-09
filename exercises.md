# Reinforcement Learning from Human Feedback (RLHF)

In this exercise, we will finetune a model pre-trained on the IMDB dataset using RLHF to generate positive reviews. 

## Introduction

### Context: Pretraining is not enough

You've seen earlier in the course that we are able to train very large and performant models like GPT2 using next-token prediction. Such models, prior to any fine-tuning, must be steered carefully with prompts in order to generate useful output. Most language models used in services of any kind today are not only pre-trained models. Rather, we use many training techniques to make them more useful. 

RLHF is one of many techniques which can convert a pre-trained model, into a more useful model for practical application.

*Context: RLHF as a naive alignment strategy*

The field AI alignment is concerned with aligning AI systems with our desired outcomes. There are many reasons to think that intelligent systems do not, by default, share human values or that whilst training against any objective will lead to reliable, expected outcomes being produced by AI systems. Nevertheless, training AI systems to produce outcomes that humans prefer over outcomes which they don't seems to be a concrete step towards AI alignment, which we can build on later. 

Thus we get the core idea of RLHF as an alignment strategy. We care about outcomes, so we provide the AI feedback based on what we think likely outcomes of it's action are and update it produce good outcomes according to our preferences. 

For more detail on RLHF, see Paul Christiano's blog post [here](https://www.alignmentforum.org/posts/vwu4kegAEZTBtpT6p/thoughts-on-the-impact-of-rlhf-research#The_case_for_a_positive_impact).

### What is RLHF

Reinforcement Learning with Human Feedback (RLHF) is a RL technique where the rewards issued by the environment are determined from a human operator. 
Often, it can be hard to specify the reward function $R : S \times A \to \mathbb{R}$ that the environment uses to issue reward to the agent, so we ask a human instead to reward/punish the agent based on the action it took. [OpenAI](https://openai.com/research/learning-from-human-preferences) uses RLHF to adjust the behaviour of models to desirable behaviour, but this can also incentivise the agent to hack the reward signal (by taking actions that look good to the human, or influencing the human to always give good rewards.)

*Why does it matter?*

RLHF (at the moment) is a successful method of nudging large language models towards desired behaviour when that behaviour is difficult to write as an algorithm.
For chess, it's easy to evaluate whether an agent won/lost the game, so we can reward that directly. For text generation, it can be hard to formally specify
that we mean by harmful or abusive text. One could have simple proxies like a filter to encourage/discourge use of particular words, and use that
to train against, but it's very easy to construct harmful text such that no particular word in the sentence would be classed as offensive:
"I would love to eat your pet puppy" contains no offensive words, even though the semantic meaning of the entire sentence is quite offensive. 
A simple proxy for offensiveness might even rate this as a positive statement, as it contains "nice" words like *love* and *puppy*.

However, samples from humans are expensive and slow. Even running a single batch of examples through the model could take a long time
if we need a human to give a scalar reward for each action chosen by the model. So, the solution is to collect a lot of data from a human
(a set of (observation, action, reward) tuples), train a reward model on this data, and then use the reward model as the reward function.

## How does RLHF work in practice

RLHF involves 3 stages:

1. We pretrain a language model (LM) using existing supervised learning techniques.
2. We gather labelled data from humans, and train a reward model that will act as a proxy for the human's rewards.
3. We fine-tuning the LM with reinforcement learning. 

### 1. Pretraining

Since reinforcement learning is very sample inefficient, it is unreasonable to expect to be able 
to train a language model from scratch using online learning. Rather, we must start with an existing 
pre-trained model and then fine-tune it. 
We will be using GPT-2-small as our base model to finetune.

![](./media/pretraining.png)


### 2. The Reward Model 

The reward model is used to assign a reward to any given output of the model during training. 
Rather than have reward be a simple function of the state of the world (as for RL environments like CartPole), 
the reward model assigns a reward to a given piece of text. 
The reward model acts like a text classifier, rewarding "good" piece of text, and punishing "bad" text.

The reward model is trained on a set of prompts, hand labelled by humans into "good" and "bad".
This is then used to train the reward model, to act as a stand-in for the human during the fine-tuning stage.

 model acts as a mapping between arbitrary text and human prefernces. 

![](media/reward-model.png)

### 3. Fine-Tuning with Reinforcement Learning 

Finally, given some reward model and some pre-trained model, we can use an algorithm such as PPO to reward the model for producing prompt completions when the reward model predicts the completion to be preferable.

In the standard RL framework, the agent recieves a reward on every timestep during interaction.
Here, the "observation" that the agent receives is a textual prompt, and the "action" the agent takes is the choice of words
to complete the prompt. The reward model then assigns a reward based on the prompt together with the completion from the agent,
which is then used to compute the loss, and update the weights of the model.


![](media/rlhf.png)

## How does RLHF differ to standard RL such as PPO?

- No "environment". RLHF operates on text completions made by the pre-trained generative model.
- Reward Model. Reward itself is generated by the reward model which itself must be trained.
- Adding a Value Head. We add a value head to the policy/LM architecture so that we have both an actor and a critic for PPO. 
- KL Divergence penalty. The KL divergence term penalizes the RL policy from moving substantially away from the initial pretrained model with each training batch, to ensure we maintain coherent outputs, and the fine-tuned model avoids generating text that overfits to what the reward model is looking for.

## Section 1: Prompt dataset and Reward model

Learning objectives
1. Load datasets from Huggingface and break them up into prompts
2. Generate text from Huggingface models 
3. Output positive sentiments from models in vanilla PyTorch and Huggingface pipelines

### Exercise: The IMDB dataset

Documentation about the IMDB dataset can be found here: https://huggingface.co/datasets/imdb. We want to use both the train and test splits to colect prompts.

Importance: 3/5
Difficulty: 1/5

if MAIN:
    imdb = load_dataset("imdb", split="train+test")

#### Exercise: Figure out the positive-negative review split in the dataset

The positive-negative review split will tell us the distribution of sentiments our model will output out of the box. Write a function to print out the number of samples for each label.

Importance: 2/5
Difficulty: 1/5

This exercise should take between 5-10 minutes

```
def label_split(dataset) -> None:
    # solution
    positive_samples = dataset['label'].count(1)
    negative_samples = dataset['label'].count(0)

    print(f"Positive reviews: {positive_samples}, Negative reviews: {negative_samples}")

if MAIN:
    n_pos, n_neg = label_split(imdb)
    tests.test_label_split(n_pos, n_neg)

```

#### Exercise: Create a set of prompts 

A prompt to the model can look like "Today was not fun ", "In the event of " or "Mary gave John a ". These prompts will serve as the starting point for model generations during the RLHF process.

In the context of the exercise to push GPT2 towards outputting reviews with more positive sentiment, we want to try and have a set of prompts that can produce varying kinds of sentiments rather than just one kind of sentiment. This set of prompts essentially forms our "observation space" and all completions are "actions", if our observation space contains primarily positive sentiment the model will not update heavily and will potentially still output negative sentiment when a prompt heavily favors it. Ideally we want our set of prompts to have a mix of sentiments.

We want to collect the first few (3-5, the choice is yours) words from each review to serve as prompts for our finetuned model. The generated text from these prompts will be later used to evaluate the performance of our finetuned model.

Importance: 3/5
Difficulty: 1/5

This exercise should take between 5-10 minutes

Hint: Use the split function to split up each review in the dataset into words

```
def generate_prompts(dataset):
    prompts = [" ".join(review.split()[:4]) for review in dataset["text"]]
    return prompts

if MAIN:
    prompts = generate_prompts(imdb)
```
### GPT-2 Finetuned on IMDB

The model that we will perform RLHF on is a GPT-2 model fine-tuned on the IMDB dataset, which can be found here: https://huggingface.co/lvwerra/gpt2-imdb. Since this model is finetuned on the IMDB dataset, the distribution of sentiments of its generations will be close to the distribution of sentiments of the original dataset. 

#### Exercise: Load the GPT-2 model and generate reviews from prompts

You will need to use the AutoTokenizer, AutoModelForCausalLM from the transformers package. You might want to use the generate method of the GPT-2 model that you load, if you do you should use top_p sampling and set the max_new_tokens argument to something that's large enough.

Play around with generating completions from this prompt and verify whether the completions approximately fit your initial expectaions of the sentiments that the model would output.

Importance: 3/5
Difficulty: 3/5

This exercise should take between 10-15 minutes
```
def generate_completion(prompt) -> str:
    
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = tokenizer.decode(model.generate(**inputs, do_sample=True, top_k=10, max_new_tokens=64).squeeze(0))
    return outputs

if MAIN:
    generate_completion(prompts[0]) 
```

### The reward function

Judging by the name of this chapter you might think that you would be providing the reward function yourself but sadly we will not be doing this. Instead, we will be using a language model trained to perform sentiment analysis to generate the sentiment score (higher is positive). The language model we will be using to generate sentiment scores can be found here: https://huggingface.co/lvwerra/distilbert-imdb. 

#### Exercise: Output sentiment scores for a generated review using vanilla PyTorch

We can use the model mentioned above in eval mode to generate sentiment scores and then transform the sentiments into rewards to be fed into the RLHF training loop.

Importance: 4/5
Difficulty: 3/5

This exercise should take between 10-15 minutes
```
def reward_model(samples, tokenizer = distilbert_tokenizer, model = distilbert_model, **kwargs) -> List[float]:

    # solution
    rewards = []

    inputs = tokenizer(samples, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    for reward in probabilities:
       rewards.append(reward[1].item())

    return rewards

if MAIN:
    example_strings = ["Example string", "I'm having a good day", "You are an ugly person"]
    rewards = reward_model(example_strings)
    tests.test_reward_model(rewards)

```

Test your model on these example strings:
example_strings = ["Example string", "I'm having a good day", "You are an ugly person"]
rewards = reward_model(example_strings)

tests.test_reward_model(rewards)

#### Exercise: Output sentiment scores for a generated review using Huggingface pipelines

This is an alternate way to get a reward model working directly using Huggingface pipelines. This will enable you to use a diverse range of models quite easily by changing a couple of arguments and provide you with more functionality than the vanilla PyTorch loop you implemented above. Reading the relevant documentation is the key to success here.

**Part A: Create a huggingface pipeline to output sentiment scores for a generated review**

Pipelines are a high-level way to use huggingface models for inference. Since the model that acts as our reward function will be used strictly for inference, it makes sense to wrap it in a pipeline. The huggingface Pipeline documentation can be found here: https://huggingface.co/docs/transformers/main_classes/pipelines

Remember to set the top_k argument to the number of labels we expect the pipeline to return, in our case this would be 2 (Positive and Negative). Also have a mechanism to ensure that the pipeline is using a gpu by setting the device argument appropriately.

We would ideally also want to use the truncation flag and the batch_size argument to enable faster generation. For this exercise, these two things are not essential but should be experimented with as we will need these for later exercises.

Importance: 4/5
Difficulty: 2/5

This exercise should take between 10-15 minutes

```
def create_pipeline(model_path):
    # solution
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

if MAIN:
    sentiment_fn = create_pipeline("lvwerra/distilbert-imdb")
```

**Part B: Map the sentiment pipeline to a reward function**

We want the reward function to return a single number corresponding to the value of the positive label (the label we care about initially) for that generation rather than a dictionary containing the labels and their respective values. 

Importance: 4/5
Difficulty: 3/5

This exercise should take between 10-15 minutes

def get_positive_score(scores):
    #solution
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

def reward_model(samples: List[str], **kwargs) -> List[float]:
    #solution
    reward = list(map(get_positive_score, sentiment_fn(samples)))
    return reward

#### Exercise: Sentiment playground

The reward model is now ready and you should take some time to feed in sentences of varying sentiments to check whether the rewards are as you expect. Remember the reward model is also a trained model so it exhibits all the quirks of one such as weird failure modes and potential to be broken with adversarial examples. 

We will also be using this opportunity to test whether your reward model is set up correctly.

Importance: 4/5
Difficulty: 1/5

This exercise should take between 5-10 minutes

```
if MAIN:
    test_prompts = ['I am happy', 'I am sad']

    rewards = reward_model(test_prompts)
    tests.test_reward_test_prompts(rewards)

    print('I want to eat', reward_model('I want to eat'))
    print('I want your puppy', reward_model('I want your puppy'))
    print('I want to eat your puppy', reward_model('I want to eat your puppy'))

## Code below has an interesting set of examples:

print('I want to eat', reward_model('I want to eat'))
print('I want your puppy', reward_model('I want your puppy'))
print('I want to eat your puppy', reward_model('I want to eat your puppy'))
```

## Section 2: Using RLHF to improve sentiment of GPT2 produced Movie Reviews

Learning objectives

### TRLX

#### What is TRLX?

trlX is a library made for training large language models using reinforcement learning. It currently supports training using PPO or [ILQL](https://arxiv.org/abs/2206.11871) for models up to 20B using Accelerate.

In practice, RLHF with trlX is very easy if you already have a reward model and pretrained model. 

#### Using trLX

Using trLX, we need to choose:

- Training Config
- A prompt dataset. 
- A reward function (which makes use of the reward model). 
- Evaluation Prompts

These 4 objects are inputs to the train function which has already been imported for you.

**Training Config**

Look below for a config that when fed into TRLX performs RLHF using PPO, all hyperparameters are set to enable training and are best left untouched for the next exercise. You might want to increase max_new_tokens to get longer generations on your evaluation prompts during finetuning. 

Increasing max_new_tokens will increase training time. For reference, keeping everything else the same in the config below and changing max_new_tokens from 40 to 100 increases finetuning time from ~6 mins to ~10 mins assuming the number of epochs and steps stay the same as the default. Picking a max_new_tokens value somewhere in the middle would be the best.

The model keyword specifies which model will be finetuned and we will be using the same GPT2 model that we used before to generate initial 

```
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
```
**Prompt Dataset**

The prompt dataset is the dataset that we'll use to generate reviews from the model specified in the config. These generations will be then be scored by the chosen reward function, this score will be used as the reward that will steer PPO to update the weights of the model towards maximising the reward function. As mentioned before the prompt dataset also forms the observation space for the PPO algorithm.

**Reward Function**

The reward function provides rewards given a set of prompt completions. In this particular case, the rewards will correspond with the positive sentiment of the completions and will steer the model towards generating strings that are generally positive .

**Evaluation Prompts**

The evaluation prompts are a set of prompts that we will use to validate the training process and the completions from these prompts will provide an indication of whether the overall sentiment is trending upwards.

We will have a single prompt repeated as the eval prompt for a number of times equal to the batch size of the reward model such as : 
```
['I am quite interested ' * batch_size_of_reward_model]
```
In this particular prompt, the initial prompt choice will cause the eval reward curve to have different starting points and end states.

## Exercise: Putting it all together - Reinforcing positive sentiment

Importance: 4/5
Difficulty: 3/5

This exercise should take between 10-15 minutes

We will now be calling the train funcation and pass in the arguments as we've described above. The train function has already been imported for you and should be called like so:

```
train(
    reward_fn = ,
    prompts = ,
    eval_prompts = ,
    config = 
) 
```

```
# provided
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

def main() -> None:
    # solution
    config = ppo_config()

    train(
        reward_fn = reward_model,
        prompts = prompts,
        eval_prompts = ['In my opinion'] * 256, ## Feel free to try different prompts
        config =  config
    )
# provided
if MAIN:
    gc.collect()
    torch.cuda.empty_cache()
    main()
```

Notice that we call torch.cuda.empty_cache() here, which is essential to free up GPU memory that might be held up as remnants of completed GPU operations or past failed runs. Running out of memory might be a common issue that you run in and running torch.cuda.empty_cache() will help you not get stuck as much. There are times when this is insufficient and you might need to restart the kernel to free up memory, you can call nvidia-smi on your terminal to see how much GPU memory is currently being used. Jupyter is unfortunately quite opaque in terms of memory management and you might need to call torch.cuda.empty_cache() and gc.collect() more often than you would expect. 

TRLX logs to W&B and you should be prompted to add in your W&B key at some point. Take a look at the reward graph that shows the change in reward received by completions from the eval_prompts over the course of the training run. All the prompt completions are stored in the files section under the media folder. 

## Exercise: Sentiment playground - Post RLHF

Try out your RLHF'd model, ideally after a gc.collect() and a torch.cuda.empty_cache() call to ensure there is free GPU memory. 

Importance: 4/5
Difficulty: 1/5

This exercise should take between 5-10 minutes

```
if MAIN:
    generate_completion('< Insert prompt here >')
```

## Exercise: Change eval prompts to observe model behaviour

Have the recurring eval_prompt be overly positive or overly negative to see the change in the reward graph. Example of a negative prompt would be - "I was extremely disappointed". Do the finetuned models at the end have different behaviours?

Importance: 2/5
Difficulty: 1/5

This exercise should take between 5-10 minutes

```
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
if MAIN:
    gc.collect()
    torch.cuda.empty_cache()
    main()
```

## Exercise: Change reward function to return high reward for neutral sentiment

Can you change the reward_fn to reinforce neutral sentiment?

Importance: 4/5
Difficulty: 3/5

```
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
if MAIN:
    gc.collect()
    torch.cuda.empty_cache()
    main()
```

## Bonus exercises

### Experiment with other huggingface models

In the above exercise, we trained a GPT2 model to output IMDB reviews with majorly positive sentiments. We can follow a similar procedure to tune other models with desirable behaviours. You can swap out the reward model pipeline and the model to be finetuned with (almost) any Huggingface model. Below are a few suggestions for tasks:

#### Fin-BERT finetuning GPT2

FinBERT is a BERT model that outputs positive and negative sentiment of financial news. The reward of outputting positive sentiment news is entangled with outputting financial news rather than any other kind of text generation. You can RLHF vanilla GPT2 with FinBERT as the reward model to verify this phenomenon and observe its effect.

FinBERT: https://huggingface.co/ProsusAI/finbert
Vanilla GPT2: https://huggingface.co/gpt2

#### Tiny stories
Doing fine-tuning with tiny stories to encourage good or bad endings of initial prompts to vanilla GPT2.

Reward model: https://huggingface.co/roneneldan/TinyStories-1M


### Calculate the KL penalty for divergence from the previous model.
Dive into the trlX trainer here: https://github.com/CarperAI/trlx/blob/404217b2f3f295ff0f68851524517064acc43a15/trlx/trainer/accelerate_ppo_trainer.py#L251

This is the function that implements many parts of the RLHF training loop, the KL divergence steps can be found starting line 430. Try replicating this code for a toy model to calculate KL divergence between this model and a copy of it during training.

### Reward Model Mechanistic interpretability  

Have a look at: https://blog.eleuther.ai/trlx-exploratory-analysis/. A great exercise could be to replicate this or find other interesting mechanistic behaviour.
