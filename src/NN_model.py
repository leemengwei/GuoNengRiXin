import torch.nn as nn
from IPython import embed
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_size, device):
        super().__init__()
        self.device = device
        self.projection = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            ).to(device)
    def forward(self, x, middle_outputs):
        # (B, H) -> (B, H)
        energy = self.projection(x)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, H) * (B, H) -> (B, H)
        outputs = middle_outputs * weights
        return outputs

class NeuralNetSimple(nn.Module):
    #This will do with:
    #--num_of_batch=10000 --hidden_width_scaler~5(any) --learning_rate=0.2 --axis_num=4
    def __init__(self, input_size, hidden_size, hidden_depth, output_size, device):
        super(NeuralNetSimple, self).__init__()
        self.attention = SelfAttention(input_size, device)
        self.fc_in = nn.Linear(input_size, hidden_size).to(device)
        self.fcs = nn.ModuleList()   #collections.OrderedDict()
        self.hidden_depth = hidden_depth
        for i in range(self.hidden_depth):
            self.fcs.append(nn.Linear(hidden_size, hidden_size).to(device))
        self.fc_middle = nn.Linear(hidden_size, input_size).to(device)
        self.fc_out = nn.Linear(input_size, output_size).to(device)
        #activations:
        self.ths = nn.Tanhshrink().to(device)
        self.sp = nn.Softplus().to(device)
        self.th = nn.Tanh().to(device)
        self.sg = nn.Sigmoid().to(device)
        self.l_relu = nn.LeakyReLU().to(device)
        self.p_relu = nn.PReLU().to(device)
    def forward(self, x):
        out = self.fc_in(x)
        for i in range(self.hidden_depth):
            out = self.fcs[i](out)
            out = self.l_relu(out)
        middle_out = self.fc_middle(out)
        #Additional attention:
        att_out = self.attention(x, middle_out)
        out = self.fc_out(att_out)
        #out = self.sp(out)
        #out = self.l_relu(out)
        out = self.p_relu(out)
        return out


