#overlap detection
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .model_tdnn import TDNN 
from .loss_functions import AngularPenaltySMLoss

# # pyannote model
# from pyannote.audio import Model
# from pyannote.audio.core.task import Task, Resolution

from pdb import set_trace as bp

class Deep_model_tdnn_ovp(nn.Module):
    def __init__(self, noclasses=2, feature_dim=40,nhid=256,dropout=0.0,pooling_function=torch.std):
        super(Deep_model_tdnn_ovp, self).__init__()
        self.tdnn1 = TDNN(input_dim=feature_dim, output_dim=nhid, context_size=5, dilation=1)
        # self.tdnn1.requires_grad = False
        self.tdnn2 = TDNN(input_dim=nhid, output_dim=nhid, context_size=3, dilation=2)
        # self.tdnn2.requires_grad = False
        self.tdnn3 = TDNN(input_dim=nhid, output_dim=nhid, context_size=3, dilation=3)
        # self.tdnn3.requires_grad = False
        self.tdnn4 = TDNN(input_dim=nhid, output_dim=nhid, context_size=1, dilation=1)
        # self.tdnn4.requires_grad = False
        self.tdnn5 = TDNN(input_dim=nhid, output_dim=128, context_size=1, dilation=1)
        # self.tdnn5.requires_grad = False


        # self.pooling_function = pooling_function
        # self.lin6 = nn.Linear(3000, 512)
        # self.bn6 = nn.BatchNorm1d(num_features=128, affine=False)
       
        # self.finlin = nn.Linear(128, noclasses)
        # self.smax = nn.Softmax(dim=1)
        # use angular softmax loss
        self.criterion = AngularPenaltySMLoss(128, noclasses, loss_type='cosface') # loss_type in ['arcface', 'sphereface', 'cosface']


        
    def prestatspool(self, x):
        
        x = F.dropout(self.tdnn1(x), p=0.5)
        x = F.dropout(self.tdnn2(x), p=0.5)
        x = F.dropout(self.tdnn3(x), p=0.5)
        x = F.dropout(self.tdnn4(x), p=0.5)
        x = F.dropout(self.tdnn5(x), p=0.5)
        # x = F.dropout(self.tdnn6(x), p=0.5)
        return x

    def forward(self, x):
        prepool = self.prestatspool(x)
        context = 7 # total is 14
        pad=(0,0,context,context) 
        out = F.pad(prepool,pad,"replicate")
        # presoftmax = self.finlin(prepool)
        return out

        
# class pyannote_gnn_model(Model):
#     def __init__(
#         self,
#         sample_rate: int = 16000, 
#         num_channels: int = 1, 
#         task: Optional[Task] = None,
#     ):
#         # First three parameters (sample_rate, num_channels, and task)
#         # must be there and passed to super().__init__()
#         super().__init__(sample_rate=sample_rate, 
#                          num_channels=num_channels, 
#                          task=task)
#         self.pynannotemodel = Model.from_pretrained("pyannote/overlapped-speech-detection", 
#                                 use_auth_token="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")
#         self.pynannotemodel.task = task

        
       