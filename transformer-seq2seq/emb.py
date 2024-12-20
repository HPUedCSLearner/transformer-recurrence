import torch
import torch.nn as nn
import math


class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, emb_size, seq_max_len):
        super().__init__()
        self.seq_max_len = seq_max_len
        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.position_embedding = torch.tensor([math.sin(pos / 1000 ** (2 * i / emb_size)) 
                                                if i % 2 == 0 else math.cos(pos / 1000 ** (2 * i / emb_size)) 
                                                for pos in range(seq_max_len) 
                                                for i in range(emb_size)]).view(seq_max_len, emb_size) # 多重循环，生成位置编码
    def forward(self, x):
        # x = [batch_size, seq_len]
        assert x.size()[1] <= self.seq_max_len, f"sequence length {x.size()[1]} exceeds the maximum sequence length {self.seq_max_len}"
        x = self.embedding(x)   # [batch_size, seq_len, emb_size]
        x = x + self.position_embedding[:x.size()[1], :].unsqueeze(0) # [batch_size, seq_len, emb_size] + [1, seq_len, emb_size]
        return x


if __name__ == '__main__':
    a = torch.tensor([i * j for i in range(10) for j in range(10, 20)]).view(10, 10)
    print(a)
    print(a.shape)
    print(a.data_ptr())

    vocab_size = 1000
    emb_size = 128
    seq_max_len = 5000

    emb = EmbeddingWithPosition(vocab_size, emb_size, seq_max_len)
    # x = torch.randint(0, 100, (32, 1000))
    x = torch.randn(32, 3333).long() % vocab_size
    print(emb(x).shape)