import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(self, in_dims=7, out_dims=1):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.fc1 = nn.Linear(self.in_dims, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, self.out_dims)
        self.dropout = nn.Dropout(0.5, inplace=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return torch.sigmoid(x)


class DeepFCNet(nn.Module):
    def __init__(self, in_dims=7, out_dims=1):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.fc1 = nn.Linear(self.in_dims, 100)
        self.bn1 = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)

        self.fc3 = nn.Linear(100, 100)
        self.bn3 = nn.BatchNorm1d(100)

        self.fc4 = nn.Linear(100, 100)
        self.bn4 = nn.BatchNorm1d(100)

        self.fc5 = nn.Linear(100, 100)
        self.bn5 = nn.BatchNorm1d(100)

        self.fc6 = nn.Linear(100, 100)
        self.bn6 = nn.BatchNorm1d(100)

        self.fc7 = nn.Linear(100, 100)
        self.bn7 = nn.BatchNorm1d(100)

        self.fc8 = nn.Linear(100, 20)
        self.bn8 = nn.BatchNorm1d(20)

        self.fc9 = nn.Linear(20, self.out_dims)
        self.dropout = nn.Dropout(0.5, inplace=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout(F.relu(self.bn4(self.fc4(x))))
        x = self.dropout(F.relu(self.bn5(self.fc5(x))))
        x = self.dropout(F.relu(self.bn6(self.fc6(x))))
        x = self.dropout(F.relu(self.bn7(self.fc7(x))))
        x = self.dropout(F.relu(self.bn8(self.fc8(x))))
        x = self.fc9(x)
        return torch.sigmoid(x)


class WideFCNet(nn.Module):
    def __init__(self, in_dims=7, out_dims=1):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.fc1 = nn.Linear(self.in_dims, 4000)
        self.bn1 = nn.BatchNorm1d(4000)
        self.fc2 = nn.Linear(4000, 4000)
        self.bn2 = nn.BatchNorm1d(4000)
        self.fc3 = nn.Linear(4000, self.out_dims)
        self.dropout = nn.Dropout(0.5, inplace=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return torch.sigmoid(x)


class CNN(nn.Module):
    def __init__(self, in_dims=7, out_dims=1):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.conv1 = nn.Conv1d(1, 8, 2)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(8, 6, 2)
        self.fc1 = nn.Linear(6, 160)  # NO CONV 3, 4
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, self.out_dims)
        self.dropout = nn.Dropout(0.5, inplace=True)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
