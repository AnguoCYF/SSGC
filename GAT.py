import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads=num_heads, feat_drop=dropout, activation=F.relu)
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, num_heads=1, feat_drop=dropout, activation=None)

    def forward(self, g, h):
        h = self.layer1(g, h).flatten(1)
        h = self.layer2(g, h).squeeze(1)
        return h

    def extract_embeddings(self, g, h):
        h = self.layer1(g, h)
        return h.view(h.size(0), -1)
