from datasets import load_dataset

dataset = load_dataset('open-r1/codeforces')

train_data = dataset['train']
test_data = dataset['test']

train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)
