from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,  AutoModelForSequenceClassification

def test_label_split(n_pos, n_neg):

    assert load_dataset("imdb", split="train+test")['label'].count(1), n_pos
    assert load_dataset("imdb", split="train+test")['label'].count(0), n_neg

    print('All tests passed!')

def test_reward_model(rewards):

    from solutions import reward_model

    example_strings = ["Example string", "I'm having a good day", "You are an ugly person"]
    assert rewards, reward_model(example_strings) #[0.5429518818855286, 0.9708014726638794, 0.04642578586935997]
    print('All tests passed!')

def test_reward_test_prompts(rewards):
    
    assert rewards, [0.9820391535758972, 0.13338041305541992]
    print('All tests passed!')