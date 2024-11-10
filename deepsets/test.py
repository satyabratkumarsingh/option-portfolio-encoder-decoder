import torch


tensor_3d = torch.tensor([[2.0, 1.5, 0.3]])
print(tensor_3d.shape)
squeezed_tensor = tensor_3d.squeeze(0)
print(squeezed_tensor.shape)
print(squeezed_tensor)