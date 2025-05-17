import re

text = "Trying to implement tokenizer. I am also learning russian.Da"
text1 = "Hi, I am doing good, i am currently implementing tokenizer. <unk>"

def preprocess_text(text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed

def load_vocab(text):
    preprocessed = preprocess_text(text)
    vocab = {token: i for i, token in enumerate(preprocessed)}
    # for i, item in enumerate(vocab.items()):
    #     print(item)
    return vocab

#load_vocab(text2)
