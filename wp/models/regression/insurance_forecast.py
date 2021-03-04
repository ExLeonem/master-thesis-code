
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class InsuranceForecast(nn.Module):
    """
        Linear regression to forecast insurance costs.
    """

    def __init__(self, input_size, output_size):
        super(InsuranceForecast, self).__init__()
        self.linear = nn.Linear(input_size, output_size)


    def forward(self, X):
        out = self.linear(X)
        return out



# Training 
input_dim = 1
output_dim = 1
learning_rate = 0.001

model = InsuranceForecast(input_dim, output_dim)
if torch.cuda.is_available():
    model.cuda()


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy)