import torch.nn as nn
import torch
from sklearn.metrics import f1_score
import math

class Head(nn.Module):
    def __init__(self, d_embedding = 128, d_hidden = 128):
        super(Head, self).__init__()
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden

        self.W_Q = nn.Linear(d_embedding, d_hidden)
        self.W_K = nn.Linear(d_embedding, d_hidden)
        self.W_V = nn.Linear(d_embedding, d_hidden)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        return torch.softmax( Q @ K.transpose(1, 2) / (self.d_hidden ** 0.5), dim = -1) @ V
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads = 8, d_embedding = 128, d_hidden = 128):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden

        self.heads = nn.ModuleList([Head(d_embedding, d_hidden) for _ in range(n_heads)])
        self.W_O = nn.Linear(n_heads * d_hidden, d_embedding)

        self.LayerNorm = nn.LayerNorm(d_embedding)

    def forward(self, x):
        attention = self.W_O(torch.cat([head(x) for head in self.heads], dim = -1))
        return self.LayerNorm(attention + x)
        
class FeedForward(nn.Module):
    def __init__(self, d_embedding = 128, d_hidden = 128):
        super(FeedForward, self).__init__()
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden

        self.W_1 = nn.Linear(d_embedding, d_hidden)
        self.W_2 = nn.Linear(d_hidden, d_embedding)

        self.LayerNorm = nn.LayerNorm(d_embedding)

    def forward(self, x):
        s = self.W_1(x)
        s = torch.relu(s)
        s = self.W_2(s)
        return self.LayerNorm(s + x)
    
class EncoderBlock(nn.Module):
    def __init__(self, d_embedding = 128, d_attention_hidden = 128, d_ffn_hidden = 128, n_heads = 8):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(n_heads, d_embedding, d_attention_hidden)
        self.ffn = FeedForward(d_embedding, d_ffn_hidden)

    def forward(self, x):
        x = self.mha(x)
        x = self.ffn(x)
        return x

class AudioClassifier(torch.nn.Module):
    def __init__(
            self, 
            n_classes, 
            d_embedding = 128, 
            n_encoder_blocks = 6,
            d_attention_hidden = 128,
            d_ffn_hidden = 128,
            n_heads = 8,
            positional_encoding = True
            ):
        super(AudioClassifier, self).__init__()
        self.positional_encoding = positional_encoding
        self.encoder_blocks = torch.nn.ModuleList([EncoderBlock(
            d_embedding = d_embedding,
            d_attention_hidden = d_attention_hidden,
            d_ffn_hidden = d_ffn_hidden,
            n_heads = n_heads
            ) for _ in range(n_encoder_blocks)])
        self.fc = torch.nn.Linear(d_embedding, n_classes)

    def forward(self, x):
        if self.positional_encoding:
            x = add_positional_encoding(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

def add_positional_encoding(x):
    batch_size, seq_len, embedding_dim = x.size()
    
    pe = torch.zeros(seq_len, embedding_dim, device=x.device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float, device=x.device) * 
                         (-math.log(10000.0) / embedding_dim))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    pe = pe.unsqueeze(0)
    return x + pe


def evaluate_f1_score(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mel_spec, labels in test_loader:
            mel_spec = mel_spec.to(device)
            labels = labels.long().to(device)

            outputs = model(mel_spec)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')  # or 'weighted', 'micro'
    return f1
        