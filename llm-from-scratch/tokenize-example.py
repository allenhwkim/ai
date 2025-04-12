import tiktoken # pip install tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Harry Potter was a wizard."

tokens = tokenizer.encode(text)

print("Num. characters", len(text), "Num. tokens", len(tokens))
print(tokens)
print(tokenizer.decode(tokens))
for t in tokens:
	print(f"{t}\t -> {tokenizer.decode([t])}")
