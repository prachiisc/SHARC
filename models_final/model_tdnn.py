import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import pickle
import subprocess
# from utils.Kaldi2NumpyUtils.kaldiPlda2numpydict import kaldiPlda2numpydict
from matplotlib import pyplot as plt
from pdb import set_trace as bp
# from model import GCNModelAE_norm
def arr2val(x, retidx):
    if x.size()[0] > 0:
        return x[retidx].cpu().item()
    else:
        return 1.

def _seconds_to_frames(t, step):
    """Convert ``t`` from seconds to frames.

    Parameters
    ----------
    t : ndarray
        Array of times in seconds.

    step : float
        Frame step size in seconds.

    Returns
    -------
    ndarray
        Frame indices corresponding to ``t``.
    """
    t = np.array(t, dtype=np.float32, copy=False)
    return np.array(t/step, dtype=np.int32)

@dataclass
class Segmentation:
    """Segmentation.

    Stores onsets/offsets of segments from a recording.

    Parameters
    ----------
    recording_id : str
        Recording segmentation is from.

    onsets: ndarray, (n_frames,)
        ``onsets[i]`` is the onset in frames of the ``i``-th segment.

    offsets: ndarray, (n_frames,)
        ``offsets[i]`` is the offset in frames of the ``i``-th segment

    step : float, optional
        Delta in seconds between onsets of consecutive frames.
        (Default: 0.01)
    """
    recording_id: str
    onsets: np.ndarray
    offsets: np.ndarray
    step: float=0.01
    
    def __post_init__(self):
        self.onsets = np.array(self.onsets, dtype=np.int32, copy=False)
        self.offsets = np.array(self.offsets, dtype=np.int32, copy=False)
        if len(self.onsets) != len(self.offsets):
            raise ValueError(
                f'"onsets" and "offsets" must have same length: '
                f'{len(self.onsets)} != {len(self.offsets)}.')
        n_bad = sum(self.durations <=0)
        if n_bad:
            raise ValueError(
                'One or more segments has non-positive duration.')

    @property
    def durations(self):
        """Segment durations in frames."""
        return self.offsets - self.onsets

    @property
    def num_segments(self):
        """Number of segments."""
        return len(self.onsets)
    
    @staticmethod
    def read_segments_file(segments_path, step=0.01):
        """Load speech segments for recordings from Kaldi ``segments`` file.

        Parameters
        ----------
        segments_path : Path
            Path to Kaldi segments file.

        step : float, optonal
            Frame step size in seconds.
            (Default: 0.01)

        Returns
        -------
        segments : dict
            Mapping from recording ids to speech segments, stored as
             ``Segmentation`` instances.
        """
        onsets = defaultdict(list)
        offsets = defaultdict(list)
        with open(segments_path, 'r') as f:
            for line in f:
                utterance_id, recording_id, onset, offset = line.strip().split()
                onsets[recording_id].append(float(onset))
                offsets[recording_id].append(float(offset))
        segments = {}
        for recording_id in onsets:
            segments[recording_id] = Segmentation(
                recording_id,
                _seconds_to_frames(onsets[recording_id], step),
                _seconds_to_frames(offsets[recording_id], step),
                step)
        return segments

class TDNN(nn.Module):

    def __init__(
            self,
            input_dim=23,
            output_dim=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=True
    ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.padlen = int(dilation * (context_size - 1) / 2)
        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        output: size (batch, new_seq_len, output_features)
        '''
        # print("In forward of TDNN")
        batch_size, _, d = tuple(x.shape)
        # print("X : ",x.shape)
        # print("D = ",d)
        # print(self.input_dim)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        # bp()

        x = F.unfold(x, (self.context_size, self.input_dim), stride=(1, self.input_dim), dilation=(self.dilation, 1))

        # N, input_dim*context_size, new_t = x.shape
        x = x.transpose(1,
                        2)  # .reshape(-1,self.context_size, self.input_dim).flip(0,1).flip(1,2).flip(1,0,2).reshape(batch_size,-1,self.context_size*self.input_dim)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x

class XVectorNet_TDNN_7Layer(nn.Module):
    def __init__(self, noclasses=17672, pooling_function=torch.std):
        super(XVectorNet_TDNN_7Layer, self).__init__()
        self.tdnn1 = TDNN(input_dim=23, output_dim=512, context_size=5, dilation=1)
        # self.tdnn1.requires_grad = False
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        # self.tdnn2.requires_grad = False
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        # self.tdnn3.requires_grad = False
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        # self.tdnn4.requires_grad = False
        self.tdnn5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        # self.tdnn5.requires_grad = False


        self.pooling_function = pooling_function
        self.lin6 = nn.Linear(3000, 512)
        self.bn6 = nn.BatchNorm1d(num_features=128, affine=False)
       
        self.finlin = nn.Linear(128, noclasses)
        self.smax = nn.Softmax(dim=1)
        
    def prestatspool(self, x):
        # bp()
        x = F.dropout(self.tdnn1(x), p=0.5)
        x = F.dropout(self.tdnn2(x), p=0.5)
        x = F.dropout(self.tdnn3(x), p=0.5)
        x = F.dropout(self.tdnn4(x), p=0.5)
        x = F.dropout(self.tdnn5(x), p=0.5)
        # x = F.dropout(self.tdnn6(x), p=0.5)
    
        return x

    def statspooling(self, x):
        average = x.mean(1)
        stddev = self.pooling_function(x,1) # x.std(1)
        concatd = torch.cat((average, stddev), 1)
        return concatd

    def postpooling(self, x):
        x = F.dropout(self.bn6(F.relu(self.lin6(x))), p=0.5)
        # x = F.dropout(self.bn7(F.relu(self.lin7(x))), p=0.5)
        x = F.relu(self.finlin(x))
        return x

    def forward(self, x):
        x = x.transpose(1, 2)
        # bp()
        # print('In forward of XvectorNet')
        prepoolout = self.prestatspool(x)
        pooledout = self.statspooling(prepoolout)
        presoftmax = self.postpooling(pooledout)
        finaloutput = self.smax(presoftmax)
        return finaloutput

    def extract(self, x):
        x = x.transpose(1, 2)
        # x = self.prestatspool(x)
        x = self.tdnn1.forward(x)
        x = self.tdnn2.forward(x)
        x = self.tdnn3.forward(x)
        x = self.tdnn4.forward(x)
        x = self.tdnn5.forward(x)
        pooledout = self.statspooling(x)
        xvec = self.lin6.forward(pooledout)
        return xvec

    def LoadFromKaldi(self, weightspath):  # Credits: Harsha Varshan
        with open(weightspath, 'rb') as f:
            kaldiweights = pickle.load(f)

        mdsd = self.state_dict()

        for i in range(1, 5):
            mdsd['tdnn{}.kernel.weight'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.affine'.format(i)]['params']).float())
            mdsd['tdnn{}.kernel.bias'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.affine'.format(i)]['bias']).float())
            mdsd['tdnn{}.bn.running_mean'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.batchnorm'.format(i)]['stats-mean']).float())
            mdsd['tdnn{}.bn.running_var'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.batchnorm'.format(i)]['stats-var']).float())

        mdsd['lin6.weight'].data.copy_(torch.from_numpy(kaldiweights['tdnn11.affine']['params']).float())
        mdsd['lin6.bias'].data.copy_(torch.from_numpy(kaldiweights['tdnn11.affine']['bias']).float())
        mdsd['bn6.running_mean'].data.copy_(torch.from_numpy(kaldiweights['tdnn11.batchnorm']['stats-mean']).float())
        mdsd['bn6.running_var'].data.copy_(torch.from_numpy(kaldiweights['tdnn11.batchnorm']['stats-var']).float())

        mdsd['finlin.weight'].data.copy_(torch.from_numpy(kaldiweights['output.affine']['params']).float())
        mdsd['finlin.bias'].data.copy_(torch.from_numpy(kaldiweights['output.affine']['bias']).float())

class XVectorNet_ETDNN_12Layer(nn.Module):
    def __init__(self, noclasses=7146, pooling_function=torch.std):
        super(XVectorNet_ETDNN_12Layer, self).__init__()
        self.tdnn1 = TDNN(input_dim=40, output_dim=1024, context_size=5, dilation=1)
        # self.tdnn1.requires_grad = False
        self.tdnn1a = TDNN(input_dim=1024, output_dim=1024, context_size=1, dilation=1)
        # self.tdnn2.requires_grad = False
        self.tdnn2 = TDNN(input_dim=1024, output_dim=1024, context_size=5, dilation=2)
        # self.tdnn3.requires_grad = False
        self.tdnn2a = TDNN(input_dim=1024, output_dim=1024, context_size=1, dilation=1)
        # self.tdnn4.requires_grad = False
        self.tdnn3 = TDNN(input_dim=1024, output_dim=1024, context_size=3, dilation=3)
        # self.tdnn5.requires_grad = False
        self.tdnn3a = TDNN(input_dim=1024, output_dim=1024, context_size=1, dilation=1)
        # self.tdnn6.requires_grad = False
        self.tdnn4 = TDNN(input_dim=1024, output_dim=1024, context_size=3, dilation=4)
        # self.tdnn7.requires_grad = False
        self.tdnn4a = TDNN(input_dim=1024, output_dim=1024, context_size=1, dilation=1)
        # self.tdnn8.requires_grad = False
        self.tdnn5 = TDNN(input_dim=1024, output_dim=2000, context_size=1, dilation=1)
        # self.tdnn9.requires_grad = False
        # self.tdnn10 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        # self.tdnn10.requires_grad = False
        self.pooling_function = pooling_function
        self.lin6 = nn.Linear(6048, 512)
        self.bn6 = nn.BatchNorm1d(num_features=512, affine=False)
        self.bn7 = nn.BatchNorm1d(num_features=512, affine=False)
        self.lin7 = nn.Linear(512, 512)
        self.finlin = nn.Linear(512, noclasses)
        self.smax = nn.Softmax(dim=1)
        
        
    def prestatspool(self, x):
        # bp()
        x = F.dropout(self.tdnn1(x), p=0.5)
        x = F.dropout(self.tdnn1a(x), p=0.5)
        x = F.dropout(self.tdnn2(x), p=0.5)
        x = F.dropout(self.tdnn2a(x), p=0.5)
        x = F.dropout(self.tdnn3(x), p=0.5)
        x = F.dropout(self.tdnn3a(x), p=0.5)
        x_4 = F.dropout(self.tdnn4(x), p=0.5)
        x = F.dropout(self.tdnn4a(x_4), p=0.5)
        x = F.dropout(self.tdnn5(x), p=0.5)
        # x = F.dropout(self.tdnn10(x), p=0.5)
        return x_4,x

    def statspooling(self, x):
        average = x.mean(1)
        stddev = self.pooling_function(x,1) # x.std(1)
        concatd = torch.cat((average, stddev), 1)
        return concatd

    def postpooling(self, x):
        x = F.dropout(self.bn6(F.relu(self.lin6(x))), p=0.5)
        x = F.dropout(self.bn7(F.relu(self.lin7(x))), p=0.5)
        x = F.relu(self.finlin(x))
        return x

    def forward(self, x):
        # x = x.transpose(1, 2)
        # bp()
        # print('In forward of XvectorNet')
        prepoolout1, prepoolout2 = self.prestatspool(x)
        pooledout1 = self.statspooling(prepoolout1)
        pooledout2 = self.statspooling(prepoolout2)
        pooledout = torch.cat((pooledout1,pooledout2),-1)
        presoftmax = self.postpooling(pooledout)
        finaloutput = self.smax(presoftmax)
        return finaloutput

    def segmented_extraction(self,x_4,x):
        '''
        From TDNN
        input: size (batch, seq_len, input_features)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
        Goal to break the sequences into 150 frames chunks with 75 shift
        output: size (batch, N_s, 150, output_features)
        Then 

        '''
        bp()
        # print("In forward of TDNN")
       
        batch_size, _, d = tuple(x_4.shape)
        # print("X : ",x.shape)
        # print("D = ",d)
        # print(self.input_dim)
        x_4 = x_4.unsqueeze(1)
        
        mydilation = 1
        mycontext_size =  150 #1.5sec
        mystride = 75 # 
        input_dim = d
       
        # Unfold input into smaller temporal contexts
        xnew = F.unfold(x_4, (mycontext_size, input_dim), stride=(mystride, input_dim), dilation=(mydilation, 1))
        xnew = xnew.transpose(1,
                            2)
        xnew = xnew.reshape(batch_size,-1,mycontext_size,input_dim)
        average = xnew.mean(2) #N,N_s,input_dim
        stddev = self.pooling_function(xnew,2) # x.std(1)
        pooledout1 = torch.cat((average, stddev), 2) # N,N_s,2*D

        batch_size, _, d = tuple(x.shape)
        input_dim = d
        x = x.unsqueeze(1)

        xnew = F.unfold(x, (mycontext_size, input_dim), stride=(mystride, input_dim), dilation=(mydilation, 1))
        # N, input_dim*context_size, N_s = x.shape
        xnew = xnew.transpose(1,
                        2)
         # N, N_s, output_dim*context_size  = x.shape
        xnew = xnew.reshape(batch_size,-1,mycontext_size,input_dim)
         # N,N_s, context_size, output_dim

        average = xnew.mean(2) #N,N_s,input_dim
        stddev = self.pooling_function(xnew,2) # x.std(1)
        pooledout2 = torch.cat((average, stddev), 2) # N,N_s,2*D

        pooledout = torch.cat((pooledout1,pooledout2),-1) # N,N_s,512
        xvec = self.lin6.forward(pooledout)
        return xvec



    def extract(self, x):
        # x = x.transpose(1, 2)
        # x = self.prestatspool(x)
        
        x = self.tdnn1.forward(x)
        x = self.tdnn1a.forward(x)
        x = self.tdnn2.forward(x)
        x = self.tdnn2a.forward(x)
        x = self.tdnn3.forward(x)
        x = self.tdnn3a.forward(x)
        x = self.tdnn4.forward(x)
        pooledout1 = self.statspooling(x)
        x = self.tdnn4a.forward(x)
        x = self.tdnn5.forward(x)
        # x = self.tdnn10.forward(x)
        
        pooledout2 = self.statspooling(x)
        pooledout = torch.cat((pooledout1,pooledout2),-1)
        xvec = self.lin6.forward(pooledout)
        return xvec

    def extract_modified(self, x):
        # x = x.transpose(1, 2)
        # x = self.prestatspool(x)
        x = self.tdnn1.forward(x)
        x = self.tdnn1a.forward(x)
        x = self.tdnn2.forward(x)
        x = self.tdnn2a.forward(x)
        x = self.tdnn3.forward(x)
        x = self.tdnn3a.forward(x)
        x_4 = self.tdnn4.forward(x)
       
        x = self.tdnn4a.forward(x_4)
        x = self.tdnn5.forward(x)
       
        return x_4,x

    def LoadFromKaldi(self, weightspath):  # Credits: Harsha Varshan
        with open(weightspath, 'rb') as f:
            kaldiweights = pickle.load(f)

        mdsd = self.state_dict()

        for i in range(1, 6):
            mdsd['tdnn{}.kernel.weight'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.affine'.format(i)]['params']).float())
            mdsd['tdnn{}.kernel.bias'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.affine'.format(i)]['bias']).float())
            mdsd['tdnn{}.bn.running_mean'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.batchnorm'.format(i)]['stats-mean']).float())
            mdsd['tdnn{}.bn.running_var'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}.batchnorm'.format(i)]['stats-var']).float())
        
        for i in range(1, 5):
            mdsd['tdnn{}a.kernel.weight'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}a.affine'.format(i)]['params']).float())
            mdsd['tdnn{}a.kernel.bias'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}a.affine'.format(i)]['bias']).float())
            mdsd['tdnn{}a.bn.running_mean'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}a.batchnorm'.format(i)]['stats-mean']).float())
            mdsd['tdnn{}a.bn.running_var'.format(i)].data.copy_(
                torch.from_numpy(kaldiweights['tdnn{}a.batchnorm'.format(i)]['stats-var']).float())


        mdsd['lin6.weight'].data.copy_(torch.from_numpy(kaldiweights['tdnn6.affine']['params']).float())
        mdsd['lin6.bias'].data.copy_(torch.from_numpy(kaldiweights['tdnn6.affine']['bias']).float())
        mdsd['bn6.running_mean'].data.copy_(torch.from_numpy(kaldiweights['tdnn6.batchnorm']['stats-mean']).float())
        mdsd['bn6.running_var'].data.copy_(torch.from_numpy(kaldiweights['tdnn6.batchnorm']['stats-var']).float())

        mdsd['lin7.weight'].data.copy_(torch.from_numpy(kaldiweights['tdnn7.affine']['params']).float())
        mdsd['lin7.bias'].data.copy_(torch.from_numpy(kaldiweights['tdnn7.affine']['bias']).float())
        mdsd['bn7.running_mean'].data.copy_(torch.from_numpy(kaldiweights['tdnn7.batchnorm']['stats-mean']).float())
        mdsd['bn7.running_var'].data.copy_(torch.from_numpy(kaldiweights['tdnn7.batchnorm']['stats-var']).float())

        mdsd['finlin.weight'].data.copy_(torch.from_numpy(kaldiweights['output.affine']['params']).float())
        mdsd['finlin.bias'].data.copy_(torch.from_numpy(kaldiweights['output.affine']['bias']).float())
        # bp()

class e2e_diarization(nn.Module):
    def __init__(self,xvec_dim=512,hidden_dim1=128,hidden_dim2=30,dropout=0,xvecmodelpath=None,device='cpu'):
        super(e2e_diarization, self).__init__()
        self.xvec_dim =xvec_dim
        self.pooling_function = torch.std
        self.xvector_extractor = XVectorNet_ETDNN_12Layer(pooling_function = self.pooling_function)
        if xvecmodelpath is not None:
            self.xvector_extractor.LoadFromKaldi(xvecmodelpath)
        # self.xvector_extractor_org = XVectorNet_ETDNN_12Layer(pooling_function = self.pooling_function)
        # self.xvector_extractor_org.LoadFromKaldi(xvecmodelpath)
        
        self.GCN_layer = GCNModelAE_norm(self.xvec_dim, hidden_dim1, hidden_dim2, dropout)
        self.device = device
        
    def train1(self):
        self.train()
        self.xvector_extractor.tdnn1.bn.training = False
        self.xvector_extractor.tdnn2.bn.training = False
        self.xvector_extractor.tdnn3.bn.training = False
        self.xvector_extractor.tdnn4.bn.training = False
        self.xvector_extractor.tdnn5.bn.training = False
       
    def forward(self,inp,adj,segmentspath,org_xvec,filenames):
        # use full MFCCs with silence
        
        n_frames = inp.shape[1]
        x_4,x = self.xvector_extractor.extract_modified(inp.unsqueeze(0)) #batch,N,D

        win = 150 #frames
        shift = 25 #frames

        # read x-vector segments file here
        frame_step=0.010
        
        
        #   recording_id, n_frames = seg.strip().split()
        
        #   n_frames = int(n_frames)

      # generate xvectors.
        # bp()
        for recording_id in filenames:
            segmentsfile = segmentspath+'/'+recording_id+'.segments'

            speech_segments = Segmentation.read_segments_file(
                segmentsfile, step=frame_step)
        #   if recording_id in speech_segments:
            segmentation = speech_segments[recording_id]
            xvecs = torch.zeros(segmentation.num_segments,self.xvec_dim).to(self.device)
            i = 0
            onset_prev = 0
            offset_prev = 0
            for onset, offset in zip(
                    segmentation.onsets, segmentation.offsets):
                if offset-onset < 100:
                    
                    diff = offset-onset
                    # print('segment dur:',diff)
                    seg1 = torch.cat((x_4[:,max(onset_prev,offset_prev-(win-diff)):offset_prev+1],x_4[:,onset:offset+1]),1)
                    seg2 =  torch.cat((x[:,max(onset_prev,offset_prev-(100-diff)):offset_prev+1],x[:,onset:offset+1]),1)
                else:
                    seg1 = x_4[:,onset:offset+1]
                    seg2 = x[:,onset:offset+1]
                pooledout1 = self.xvector_extractor.statspooling(seg1)
                pooledout2 = self.xvector_extractor.statspooling(seg2)

                pooledout = torch.cat((pooledout1,pooledout2),-1)
                xvecs[i] = self.xvector_extractor.lin6.forward(pooledout).squeeze(0)
                i = i+1
                onset_prev = onset
                offset_prev = offset
            x_new = 0.5*(xvecs + org_xvec)
            z = self.GCN_layer(x_new,adj)
        
        return z


class e2e_diarization_nosilence(nn.Module): # remove silence from features before feeding 

    def __init__(self,xvec_dim=512,hidden_dim1=128,hidden_dim2=30,dropout=0,xvecmodelpath=None,device='cpu'):
        super(e2e_diarization_nosilence, self).__init__()
        self.xvec_dim =xvec_dim
        self.pooling_function = torch.std
        self.xvector_extractor = XVectorNet_ETDNN_12Layer(pooling_function = self.pooling_function)
       
        if xvecmodelpath is not None:
            self.xvector_extractor.LoadFromKaldi(xvecmodelpath)
        # self.xvector_extractor.eval()
        # self.xvector_extractor_org = XVectorNet_ETDNN_12Layer(pooling_function = self.pooling_function)
        # self.xvector_extractor_org.LoadFromKaldi(xvecmodelpath)
        
        self.GCN_layer = GCNModelAE_norm(self.xvec_dim, hidden_dim1, hidden_dim2, dropout)
        self.device = device
        
    def train1(self):
        self.train()
        self.xvector_extractor.tdnn1.bn.training = False
        self.xvector_extractor.tdnn1a.bn.training = False
        self.xvector_extractor.tdnn2.bn.trainsing = False
        self.xvector_extractor.tdnn2a.bn.training = False
        self.xvector_extractor.tdnn3.bn.training = False
        self.xvector_extractor.tdnn3a.bn.training = False
        self.xvector_extractor.tdnn4.bn.training = False
        self.xvector_extractor.tdnn4a.bn.training = False
        self.xvector_extractor.tdnn5.bn.training = False
       
    def mfcc_full_xvec_segmented(self,inp,adj,segmentsfile,org_xvec,filenames):
        # use full MFCCs without silence and create segments using segments file
        subsample = 1 # subsampling factor for x-vector 
        n_frames = inp.shape[1]
        x_4,x = self.xvector_extractor.extract_modified(inp.unsqueeze(0)) #batch,N,D

        win = 150 #frames
        shift = 25 #frames

        # read x-vector segments file here
        frame_step=0.010
        
        #   recording_id, n_frames = seg.strip().split()
        #   n_frames = int(n_frames)
        # generate xvectors.
        # bp()
        count = 0
        for idx,recording_id in enumerate(filenames):
        #   if recording_id in speech_segments:
            segmentation = segmentsfile[idx]
            xvecs = []

            segend = 0
            for onset, offset in zip(
                    segmentation.onsets, segmentation.offsets):
                bp()
                segdiff = offset-onset
                start = segend
                end = start + win
                diff = offset-onset
                if diff < 150:
                    if diff > 50:
                        
                        end  = min(end,segend+segdiff)
                        if count % subsample ==0:
                            seg1 = x_4[:,start:end]
                            seg2 = x[:,start:end]
                            pooledout1 = self.xvector_extractor.statspooling(seg1)
                            pooledout2 = self.xvector_extractor.statspooling(seg2)
                            pooledout = torch.cat((pooledout1,pooledout2),-1)
                            xvecs.append(self.xvector_extractor.lin6.forward(pooledout))
                        count = count + 1
                else:
                    
                    while end < segend + segdiff:
                        # if  segend + segdiff - end <= offset: 
                        #     end = segend + segdiff
                        if count % subsample ==0:
                            seg1 = x_4[:,start:end]
                            seg2 = x[:,start:end]
                            pooledout1 = self.xvector_extractor.statspooling(seg1)
                            pooledout2 = self.xvector_extractor.statspooling(seg2)

                            pooledout = torch.cat((pooledout1,pooledout2),-1)
                            xvecs.append(self.xvector_extractor.lin6.forward(pooledout))
                            
                        start = start + shift
                        end = start + win
                        count = count + 1
                    if end >= segend + segdiff:
                        end =  segend + segdiff
                        start = end - win
                        if count % subsample ==0:
                            seg1 = x_4[:,start:end]
                            seg2 = x[:,start:end]
                            pooledout1 = self.xvector_extractor.statspooling(seg1)
                            pooledout2 = self.xvector_extractor.statspooling(seg2)

                            pooledout = torch.cat((pooledout1,pooledout2),-1)
                            xvecs.append(self.xvector_extractor.lin6.forward(pooledout))
                        count = count + 1
                
                segend = segend + segdiff
            
            xvecs = torch.cat(xvecs,dim=0)
            bp()
            x_new = 0.5*(xvecs + org_xvec)
            z = self.GCN_layer(x_new,adj)
        
        return z
    

    def mfcc_full_moving_avg(self,inp,filenames):
        # use full MFCCs without silence and use moving average window function calculate stats and xvectors
        bp()
        subsample = 1 # subsampling factor for x-vector 
        n_frames = inp.shape[1]
        x_4,x = self.xvector_extractor.extract_modified(inp.unsqueeze(0)) #batch,N,D
        win = 150 #frames
        shift = 75 #frames
        xvecs = self.xvector_extractor.segmented_extraction(x_4,x)
        return xvecs


    def mfcc_segmented(self,inp,adj,org_xvec,segmentsfile=None,filenames=None):
        # use MFCCs without silence in batches already
        fullbatchsize,n_frames,D = inp.shape
        batchsize = min(1000,fullbatchsize) 
        batch_count = int(fullbatchsize/batchsize)
        cur_batch = batchsize*batch_count
        remainder = fullbatchsize - cur_batch
        xvecs = []
       
        inp_new = inp[:batchsize*batch_count].reshape(batch_count,batchsize,n_frames,D)
        
        for i in range(batch_count):
            xvecs.append(self.xvector_extractor.extract(inp_new[i])) #batch,N,D
            print('count:',i)
            
        if remainder > 0:
            xvecs.append(self.xvector_extractor.extract(inp[cur_batch:]))
        
        xvecs = torch.cat(xvecs,dim=0)
        # bp()
        # x_new = 0.5*(xvecs + org_xvec)
        # z = self.GCN_layer(x_new,adj)
        
        return xvecs
