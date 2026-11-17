"""Exact imitation initialization for the SAME published direct-output MLP.

All weights remain trainable. There is no skip connection, residual addition,
reference-relative clamp, or postprocessing that restores the reference.
"""
import torch
from pi05_direct_networks import DirectActor

class IdentityDirectActor(DirectActor):
    def __init__(self,zmean=None,zstd=None,smean=None,sstd=None):
        super().__init__(zmean,zstd,smean,sstd)
        self.double()
        self.model.ref_dropout=0.
        first,second,last=self.model.net[0],self.model.net[2],self.model.net[4]
        # ReLU(x)-ReLU(-x)=x encoded as ordinary, trainable MLP weights.
        # Unused hidden units retain random initialization; their output weights
        # start at zero and can learn contextual features through Q gradients.
        with torch.no_grad():
            first.weight[:70].zero_();first.bias[:70].zero_()
            second.weight[:70].zero_();second.bias[:70].zero_()
            last.weight.zero_();last.bias.zero_()
            for i in range(35):
                first.weight[i,264+i]=1.;first.weight[35+i,264+i]=-1.
                second.weight[i,i]=1.;second.weight[35+i,35+i]=1.
                last.weight[i,i]=1.;last.weight[i,35+i]=-1.
