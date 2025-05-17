import tiktoken

text = "Hi, I am doing good, i am currently implementing tokenizer. <unk>"

# Here we are using gpt2 tokenizer, you could use llama if you want

tokenizer = tiktoken.get_encoding("gpt2")
encoding_bp = tokenizer.encode(text)
print(encoding_bp)
