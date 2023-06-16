import torch.nn as nn


class NetworkModel(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, num_classes):
        super(NetworkModel, self).__init__()
        self.layer1 = nn.Linear(input_dimension, hidden_dimension)
        self.layer2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.layer3 = nn.Linear(hidden_dimension, num_classes)
        self.activation = nn.ReLU()

    def forward(self, input_data):
        output = self.layer1(input_data)
        output = self.activation(output)
        output = self.layer2(output)
        output = self.activation(output)
        output = self.layer3(output)
        return output
