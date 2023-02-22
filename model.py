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


class CustomLSTM(torch.nn.Module):
    # https://www.kaggle.com/code/andradaolteanu/pytorch-rnns-and-lstms-explained-acc-0-99
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layer_size: int,
        num_classes: int,
        device: torch.device,
        batch_size: int = 1,
        bidirectional: bool = False,
    ):
        super(CustomLSTM, self).__init__()
        self.input_size, self.hidden_size, self.layer_size, self.output_size = input_size, hidden_size, layer_size, num_classes
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.device = device

        self.lstm = torch.nn.LSTM(
            input_size*input_size,
            hidden_size,
            layer_size,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        # Create FNN
        if self.bidirectional:
            self.fnn = torch.nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fnn = torch.nn.Linear(hidden_size, num_classes)

        if self.bidirectional:
            self.hidden_state = torch.zeros(
                self.layer_size * 2,
                self.batch_size,
                self.hidden_size,
            ).to(self.device)
            self.cell_state = torch.zeros(
                self.layer_size * 2,
                self.batch_size,
                self.hidden_size,
            ).to(self.device)
        else:
            self.hidden_state = torch.zeros(
                self.layer_size,
                self.batch_size,
                self.hidden_size,
            ).to(self.device)
            self.cell_state = torch.zeros(
                self.layer_size,
                self.batch_size,
                self.hidden_size,
            ).to(self.device)


    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        output, _ = self.lstm(x, (self.hidden_state, self.cell_state))

        # FNN
        output = self.fnn(output[:, -1, :])
        return output
