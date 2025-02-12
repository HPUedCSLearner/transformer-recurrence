import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_MAX_LEN = 5000