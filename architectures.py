import torch
import torch.nn as nn
import torch.nn.functional as F

"""
24C5 means a convolution layer with 24 feature maps using a 5x5 filter and stride 1
24C5S2 means a convolution layer with 24 feature maps using a 5x5 filter and stride 2
P2 means max pooling using 2x2 filter and stride 2
256 means fully connected dense layer with 256 units
"""

cfg = {
    'net_1': [32, 'MaxPool'],
    'net_1_double': [32, 32, 'MaxPool'],
    'net_1_triple': [32, 32, 32, 'MaxPool'],
    'net_2': [32, 'MaxPool', 64, 'MaxPool'],
    'net_2_double': [32, 32, 'MaxPool', 64, 64, 'MaxPool'],
    'net_2_triple': [32, 32, 32, 'MaxPool', 64, 64, 64, 'MaxPool'],
    'net_3': [32, 'MaxPool', 64, 'MaxPool', 96, 'MaxPool'],
    'net_3_double': [32, 32, 'MaxPool', 64, 64, 'MaxPool', 128, 128, 'MaxPool'],
    'net_3_triple': [32, 32, 32, 'MaxPool', 64, 64, 64, 'MaxPool', 128, 128, 128, 'MaxPool']
}

class PlaNet(nn.Module):
    def __init__(self, net_name, batch_norm = True, input_size = (1,2,128,128)):
        super(PlaNet, self).__init__()
        self.features = self._make_feature_extractor(net_name, batch_norm, input_size)
        flatten_features_size = self.num_flat_features(self.features(torch.rand(input_size)))
        self.classifier = nn.Sequential(
            nn.Linear(flatten_features_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 29),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.5)
        print(' ')

    def forward(self, x):
        # extract features
        x = self.features(x)
        # dropout
        x = self.dropout(x)
        # flatten
        x = x.view(-1, self.num_flat_features(x))
        # classify
        x = self.classifier(x)
        return x

    def _make_feature_extractor(self, net_name, batch_norm, input_size):
        layers = []
        in_channels = input_size[1]
        layer_key_list = cfg[net_name]
        for l in layer_key_list:
            if l == 'MaxPool':
                layers += [nn.MaxPool2d(2, 2)]
            else:
                layers += [nn.Conv2d(in_channels, l, 3, bias=False)]
                if batch_norm:
                    layers += [nn.BatchNorm2d(l), nn.ReLU(inplace=True)]
                else:
                    layers += [nn.ReLU(inplace=True)]
                in_channels = l
        return nn.Sequential(*layers)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class MLP_1(nn.Module):
    """
    (1x128x128) => 1024-RLU => 256-RLU => 29
    # [20,   250] loss: 0.528
    """
    def __init__(self):
        super(MLP_1, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1 = nn.Linear(1 * 128 * 128, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 29)

    def forward(self, x):
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # (1x128x128) = > 256 - RLU
        x = self.fc1(x)
        x = F.relu(x)
        #  (256) = > 128 - RLU
        x = self.fc2(x)
        x = F.relu(x)
        # (128) => 29 - SOFTMAX
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_1(nn.Module):
    """
    24C3-RLU-P2 => 256-RLU => 29-SIG
    [20, 550] loss:   0.2604, accuracy: 0.776
    """
    def __init__(self):
        super(CNN_1, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 24, 3, bias=False)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(24 * 63 * 63, 256)
        self.fc2 = nn.Linear(256, 29)

    def forward(self, x):
        # 24C3-RLU-P2
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # 256-RLU
        x = self.fc1(x)
        x = F.relu(x)
        # 29-SIG
        x = self.fc2(x)
        return torch.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_2(nn.Module):
    """
    24C3-RLU-P2 => 48C3-RLU-P2 => 256-RLU => 29-SIG
    [20, 550] loss:   0.3014, accuracy: 0.783
    """
    def __init__(self):
        super(CNN_2, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 24, 3, bias=False)
        self.conv2 = nn.Conv2d(24, 48, 3, bias=False)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(48 * 30 * 30, 256)
        self.fc2 = nn.Linear(256, 29)

    def forward(self, x):
        # 24C5-RLU - P2
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # - 48C5-RLU - P2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # 256-RLU - 29-SMAX
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_3(nn.Module):
    """
    24C3-RLU-P2 => 48C3-RLU-P2 => 96C3-RLU-P2 => 256-RLU => 29-SIG
    [20, 550] loss:   0.3556, accuracy: 0.759
    """
    def __init__(self):
        super(CNN_3, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 24, 3, bias=False)
        self.conv2 = nn.Conv2d(24, 48, 3, bias=False)
        self.conv3 = nn.Conv2d(48, 96, 3, bias=False)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(96 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 29)

    def forward(self, x):
        # 24C3-RLU-P2
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # 48C3-RLU-P2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # 96C3-RLU-P2
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # 256-RLU
        x = self.fc1(x)
        x = F.relu(x)
        # 29-SIG
        x = self.fc2(x)
        return torch.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNNt_1(nn.Module):
    """
    24C3-RLU-P2 => 256-RLU => 29-SIG
    no fft: [20, 550] loss:   0.2604, accuracy: 0.776
    [20, 550] loss:   0.3654, accuracy: 0.764
    """
    def __init__(self):
        super(CNNt_1, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 24, 3, bias=False)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(24 * 63 * 63, 256)
        self.fc2 = nn.Linear(256, 29)

    def forward(self, x):
        # 24C3-RLU-P2
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # 256-RLU
        x = self.fc1(x)
        x = F.relu(x)
        # 29-SIG
        x = self.fc2(x)
        return torch.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNNt_2(nn.Module):
    """
    24C3-RLU-P2 => 48C3-RLU-P2 => 256-RLU => 29-SIG
    w/o ft: [20, 550] loss:   0.3014, accuracy: 0.783
    [20, 550] loss:   0.2456, accuracy: 0.761 (optimizer = Adam with lr=0.0001, betas=(0.9, 0.999))
    """
    def __init__(self):
        super(CNNt_2, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 24, 3, bias=False)
        self.conv2 = nn.Conv2d(24, 48, 3, bias=False)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(48 * 30 * 30, 256)
        self.fc2 = nn.Linear(256, 29)

    def forward(self, x):
        # 24C3-RLU-P2
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # 48C3-RLU-P2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # FLATTEN
        x = x.view(-1, self.num_flat_features(x))
        # 256-RLU - 29-SMAX
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNNt_2a(nn.Module):
    """
    24C3-RLU-P2 => 48C3-RLU-P2 => 256-RLU => 29-SIG
    w/o ft: [20, 550] loss:   0.3014, accuracy: 0.783
    ft: [20, 550] loss:   0.2456, accuracy: 0.761 (optimizer = Adam with lr=0.0001, betas=(0.9, 0.999))

    """
    def __init__(self):
        super(CNNt_2a, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.features = nn.Sequential(
            nn.Conv2d(2, 24, 3, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(24),
            # 48C3-RLU-P2
            nn.Conv2d(24, 48, 3, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),
        )
        self.classifier = nn.Sequential(
            nn.Linear(48 * 30 * 30, 256),
            nn.ReLU(True),
            nn.Linear(256, 29),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # extract features
        x = self.features(x)
        # dropout
        x = self.dropout(x)
        # flatten
        x = x.view(-1, self.num_flat_features(x))
        # classify
        x = self.classifier(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
