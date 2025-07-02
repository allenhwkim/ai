import torch
from torch.utils.data import DataLoader, TensorDataset

data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)

dataset = TensorDataset(data)  # Wrap data in a TensorDataset
dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

for batch in dataloader:
    batch_data = batch[0]  # Extract the batch tensor
    print(batch_data) 
    # tensor([1., 2., 3.])
    # tensor([4., 5., 6.])
    # tensor([7., 8., 9.])
    # tensor([10.])