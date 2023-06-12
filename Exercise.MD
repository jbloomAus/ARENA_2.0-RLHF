# Reinforcement Learning from Human Feedback (RLHF)

In this exercise, we will finetune a model pre-trained on the IMDB datasert using RLHF to generate positive reviews. 

## The IMDB dataset

Documentation about the IMDB dataset can be found here: https://huggingface.co/datasets/imdb. 

### Exercise: Figure out the positive-negative review split in the dataset

Based on your above results, you should get an idea of the likelihood of a model finetuned on this dataset to generate a positive review. 

### Exercise: Create a set of prompts 

We want to collect the first few (3-5, the choice is yours) words from each review to serve as prompts for our finetuned model. The generated text from these prompts will be later used to evaluate the performance of our finetuned model.

## GPT-2 Finetuned on IMDB

The model that we will perform RLHF on is a GPT-2 model fine-tuned on the IMDB dataset, which can be found here: https://huggingface.co/lvwerra/gpt2-imdb. Since this model is finetuned on the IMDB dataset, the distribution of sentiments of its generations will be close to the distribution of sentiments of the original dataset. 

### Exercise: Load the GPT-2 model and generate reviews from prompts

You will need to use the AutoTokenizer, AutoModelForCausalLM from the transformers package. You might want to use the generate method of the GPT-2 model that you load, if you do you should use top_p sampling and set the max_new_tokens argument to something that's large enough.

## The reward function

Judging by the name of this chapter you might think that you would be providing the reward function yourself but sadly we will not be doing this. Instead, we will be using a language model trained to perform sentiment analysis to generate the sentiment score (higher is positive). The language model we will be using to generate sentiment scores can be found here: https://huggingface.co/lvwerra/distilbert-imdb. 

### Exercise: Create a huggingface pipeline to outputs sentiment scores for a generated review

Pipelines are a high-level way to use huggingface models for inference. Since the model that acts as our reward function will be used strictly for inference, it makes sense to wrap it in a pipeline. The huggingface Pipeline documentation can be found here: https://huggingface.co/docs/transformers/main_classes/pipelines

Remeber to set the top_k argument to the number of labels we expect the pipeline to return, in our case this would be 2 (Positive and Negative). 

We would ideally also want to use the truncation flag and the batch_size argument to enable faster generation. For this exercise, these two things are not essential but should be experimented with as we will need these for later exercises.

### Exercise: Map the sentiment pipeline to a reward function

We want the reward function to return a single number corresponding to the value of the positive label for that generation rather than a dictionary containing the labels and their respective values. 

## Positive IMDB reviews: Putting it all together


## GPT-2 positive sentiment completions 


## Bonus exercises

### Experiment with other huggingface models

### Get maximally "positive" string according to the RLHF'd model


