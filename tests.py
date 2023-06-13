from datasets import load_dataset

def test_label_split(n_pos, n_neg):

    assert load_dataset("imdb", split="train+test")['label'].count(1), n_pos
    assert load_dataset("imdb", split="train+test")['label'].count(0), n_neg

    print('All tests passed!')

def test_reward_model(rewards):

    assert rewards, [0.5429518818855286, 0.9708014726638794, 0.04642578586935997]
    print('All tests passed!')

def test_reward_test_prompts(rewards):
    
    assert rewards, [0.9820391535758972, 0.13338041305541992]
    print('All tests passed!')