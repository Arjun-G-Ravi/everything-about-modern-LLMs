'''
I'm trying to code a very simple decoder only llm
'''
from transformers import AutoTokenizer
import torch
import torch.nn as nn

context_length = 512
embedding_dim = 768
num_transformer_layers = 8
 
class TransformerBlock:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return x

class MLPBlock:
    def __init__(self):
        self.lay1 = nn.Linear(768)

def llm(x):
    tokenizer = AutoTokenizer.from_pretrained("gpt2") 
    token_ids = torch.tensor(tokenizer.encode(x))
    embedding = nn.Embedding(num_embeddings=50304, embedding_dim=embedding_dim)
    pos_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dim)

    positions = torch.arange(len(token_ids))

    embedded = embedding(token_ids)
    pos_embedded = pos_embedding(positions)
    final_embeddings = embedded + pos_embedded

    print(final_embeddings)

    t_block = TransformerBlock()
    out = final_embeddings
    for _ in range(num_transformer_layers):
        out = t_block(out)
    print(out.shape)

    
    

       









if __name__ == '__main__':
    llm('what is your name?')