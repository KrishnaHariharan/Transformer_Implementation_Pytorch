import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    '''
    In the original research paper standard embedding 
    is multiplied by square root of the dimension
    '''
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self,d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #Matrix of Shape (Seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        #Creaet a vector of shape(seq_len, 1)
        '''
        Arrange - Returns a tensor with values from interval start to end
                  By default the interval/ step to be 1
        
        Unsqueeze - Adds an extra dimension of size 1 at the point you want 
        '''
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #Apply sine and cosine to the even and odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        self.register_buffer('pe', pe) #To save this parameter while saving the model apart from learnt parameters

    
    def forward(self, x):
        #Since we dont want to learn this positional encoding hence requires)grad
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
        





        

