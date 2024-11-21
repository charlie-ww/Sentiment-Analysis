
import pickle
import sys

class Tokenizer:
    def __init__(self, vocab='token_to_index.pkl', max_length=64):
        """
        Tokenizer class that encodes text to tokenized index vector and decodes tokenized index vector to text.
        
        Args:
            vocab (str, optional): Defaults to 'token_to_index.pkl'. The path to load token_to_index dictionary.
            max_length (int, optional): Defaults to 64. The maximum length of the tokenized index vector.
            
        """
        # load the vocabulary from pickle file (dictionary that maps tokens to indices)
        with open(vocab, 'rb') as f:    
            self.vocab = pickle.load(f)
        
        # define special tokens, TODO: you can re-define special tokens if needed
        self.bos_token_id = self.vocab['[BOS]']
        self.eos_token_id = self.vocab['[EOS]']
        self.pad_token_id = self.vocab['[PAD]']
        self.unk_token_id = self.vocab['[UNK]']
        
        self.max_length = max_length
        self.id_to_token = {token_id: token for token, token_id in self.vocab.items()}
 
 
    def encode(self, text):
        
        # 1. split sentence into tokens 
        text = text.lower()  
        tokens = text.split(' ')  # TODO: make sure you split text in the same way as you did when creating the dictionary

        
        # 2. convert tokens to vector of indices
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        
        # 3. append [BOS] and [EOS] tokens at the beginning and end of token_ids
        token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        

        # 4. truncate or pad token_ids with pad_token to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids += [self.pad_token_id] * (self.max_length - len(token_ids))  
        return token_ids


    def batch_encode(self, texts):
        return [self.encode(text) for text in texts]


    def get_vocab_size(self):
        return len(self.vocab)


if __name__ == "__main__":
    # TODO: examples of how to use the Tokenizer class
    example_texts = ["Embrace self-love and simplicity in life.",
                     "The app offers a wide range of features, which is impressive."
                     ]

    tokenizer = Tokenizer(vocab='token_to_index.pkl', max_length=16)
    tokenized_text = tokenizer.batch_encode(example_texts)
    for text, token_ids in zip(example_texts, tokenized_text):
        print(f"\n[Text]: {text}")
        print(f"[Token_ids]: {token_ids}")
        decoded = " ".join([tokenizer.id_to_token.get(token_id, " ") for token_id in token_ids])
        print(f"[Decode_text]: {decoded}")

        
  