from vocab import load_vocab, preprocess_text

vocab = load_vocab("Hi, I am doing good, i am currently implementing tokenizer. <unk>")

class simpletokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = preprocess_text(text)
        preprocessed = [items if items in self.str_to_int else "<unk>" for items in preprocessed]
        encoded_text = [self.str_to_int[s] for s in preprocessed]
        return encoded_text

text1 = "how am i"
tokenizer = simpletokenizer(vocab)
val = tokenizer.encode(text1)
print(val)
