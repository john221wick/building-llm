from vocab import load_vocab, preprocess_text

text = "Trying to implement tokenizer. I am also learning russian.Da"
vocab = load_vocab(text)

class simpletokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = preprocess_text(text)
        encoded_val =  [self.str_to_int[s] for s in preprocessed]
        return encoded_val
    def decode(self, encoded_val):
        decoded_val = [self.int_to_str[i] for i in encoded_val]
        return decoded_val

tokenizer = simpletokenizer(vocab)
text1 = "I am"
# Encoding
val = tokenizer.encode(text1)
print(val)
# Decoding
print(tokenizer.decode(val))
