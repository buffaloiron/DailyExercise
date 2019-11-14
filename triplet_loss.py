import torch
import torch.nn as nn

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

input1 = torch.randn(1, 3, requires_grad=True)
input2 = torch.randn(1, 3, requires_grad=True)
input3 = torch.randn(1, 3, requires_grad=True)
input1.retain_grad()
input2.retain_grad()
input3.retain_grad()

print(input1,input2,input3)

output = triplet_loss(input1, input2, input3)
output.retain_grad()

print(output)

output.backward()

print('grads:',output.grad,input1.grad,input2.grad,input3.grad)
