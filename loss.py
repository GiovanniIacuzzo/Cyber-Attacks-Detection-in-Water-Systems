# losses.py

import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, softmax_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.softmax_weight = softmax_weight

    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)
        softmax_reg = torch.mean(torch.softmax(outputs, dim=1))
        loss = self.ce_weight * ce + self.softmax_weight * (1.0 - softmax_reg)
        return loss
