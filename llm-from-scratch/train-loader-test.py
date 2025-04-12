from train_loader import train_loader
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.n_vocab)

# Iterating Through DataLoader
dataiter = iter(train_loader)
x, y = next(dataiter)

# shuffle=True
print(tokenizer.decode(x[0].tolist()))
print(tokenizer.decode(y[0].tolist()))
print(tokenizer.n_vocab)
