import numpy as np
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
np_array_data = torch.tensor(np_array)

x_ones = torch.ones_like(x_data)
print(x_ones)
x_random = torch.rand_like(x_data, dtype=torch.float)
print(x_random)

shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

print(
    f"Atributos del tensor: \n Shape:{rand_tensor.shape} \n Type: {rand_tensor.dtype} \n Device: {rand_tensor.device}"
)

if torch.accelerator.is_available():
    rand_tensor = rand_tensor.to(device=torch.accelerator.current_accelerator())
print(f"Device: {rand_tensor.device}")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

t1 = torch.cat((tensor, tensor, tensor), dim=1)
print(t1)

y1 = tensor @ tensor.T
print(y1)
y2 = torch.matmul(tensor, tensor.T) 
print(y2)

z1 = tensor * tensor
print(z1)
z2 = tensor.mul(tensor)
print(z2)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

n = tensor.numpy()
tensor.add_(5)
print(n)