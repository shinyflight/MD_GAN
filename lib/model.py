import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import uneye
import yaml

with open("MD_GAN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size+1, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, z, y):
        # process y
        y = uneye(y, 'model')
        # concat condition
        z = torch.cat((z,y), 1)
        # feedforward
        z = F.relu(self.bn(self.fc1(z)))
        G_z = F.sigmoid(self.fc2(z))
        return G_z


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.out_gan = nn.Linear(hidden_size, output_size)
        self.out_aux = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        D_x = F.sigmoid(self.out_gan(x))
        C = F.softmax(self.out_aux(x))
        return D_x, C


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)