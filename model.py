import torch


class CustomFCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomFCN, self).__init__()
        self.input_size = input_size
        self.flatten = torch.nn.Flatten()
        self.layer_1 = torch.nn.Linear(input_size, hidden_size) 
        self.relu_1 = torch.nn.ReLU()
        self.dropout_1 = torch.nn.Dropout(0.2)
        self.layer_2 = torch.nn.Linear(hidden_size, hidden_size) 
        self.relu_2 = torch.nn.ReLU()
        self.dropout_2 = torch.nn.Dropout(0.2)
        self.layer_3 = torch.nn.Linear(hidden_size, hidden_size) 
        self.relu_3 = torch.nn.ReLU()
        self.dropout_3 = torch.nn.Dropout(0.2)
        self.layer_4 = torch.nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x)
        x = self.layer_3(x)
        x = self.relu_3(x)
        x = self.dropout_3(x)
        out = self.layer_4(x)
        # no activation and no softmax at the end
        return out

class CustomCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 12, 5)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(12, 24, 5)
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.conv4 = torch.nn.Conv2d(24, 1, 5)
        self.fc1 = torch.nn.Linear(400, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool2(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool3(torch.nn.functional.relu(self.conv3(x)))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
