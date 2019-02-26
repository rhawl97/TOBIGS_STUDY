from configs import src

import torch.nn as nn
import torch.nn.functional as F

class TobigsNet(nn.Module):
    def __init__(self, src):
        super(TobigsNet, self).__init__()
        self.fc1 = nn.Linear(src['input_size'], src['hidden_size1'])
        self.fc2 = nn.Linear(src['hidden_size1'], src['hidden_size2'])
        self.fc3 = nn.Linear(src['hidden_size2'], src['hidden_size3'])
        self.fc4 = nn.Linear(src['hidden_size3'], src['output_size'])

        ## sequential layer
        self.seq_fc = nn.Sequential(
                            nn.Linear(src['input_size'], src['hidden_size1']),
                            nn.Linear(src['hidden_size1'], src['hidden_size2']),
                            nn.Linear(src['hidden_size2'], src['hidden_size3']),
                            nn.Linear(src['hidden_size3'], src['output_size'])
                            )

        self.init_range = src['init_weight_range']

    def init_weight(self):
        self.fc1.weight.data.uniform_(-self.init_range, self.init_range)
        self.fc2.weight.data.uniform_(-self.init_range, self.init_range)
        self.fc3.weight.data.uniform_(-self.init_range, self.init_range)
        self.fc4.weight.data.uniform_(-self.init_range, self.init_range)

        for fc in self.seq_fc:
            fc.weight.data.uniform_(-self.init_range, self.init_range)

    def forward(self, img):
        x = img.view(img.shape[0], -1)
        #--------------------
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        y = self.fc4(x)
        #--------------------

        ## 4 lines above are identical to
#         y = self.seq_fc(x)

        return y

model = TobigsNet(src)
