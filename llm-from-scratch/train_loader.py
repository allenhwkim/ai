# pip install torch tiktoken
import torch 
from torch.utils.data import Dataset, DataLoader
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

class MyDataset(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        print("# of tokens in txt:", len(token_ids))

        for i in range(0, len(token_ids) - max_length, stride):
		    # If max_length is set to 32, each input_chunk and 
            # target_chunk will contain 32 tokens. This ensures 
            # that the model processes sequences of a consistent 
            # length, which is important for training.
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

		# Retrieves a sample by index.
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

with open("harry-potter-02.txt.cleaned", "r", encoding="utf-8-sig") as file:
    txt = file.read()

dataset = MyDataset(txt, max_length=32, stride=4)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
