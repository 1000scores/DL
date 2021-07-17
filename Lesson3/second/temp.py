import torch
a = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
b = torch.tensor([5.0], requires_grad=True)
c = a * b
c.backward()
print(a.grad)
c = c * b
c.backward()
print(a.grad)