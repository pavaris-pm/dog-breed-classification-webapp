from torch import nn
import torch


class ConvNextTransfer(nn.Module):
  '''Transfer learning of ConvNext Architecture'''
  
  def __init__(self, 
               convnext_block, 
               num_labels: int,
               ):
    
    super().__init__()
    self.convnext = convnext_block
    # this is for predicting dog breed
    self.classifier = nn.Linear(in_features=1024, 
                                out_features=num_labels)
      

  def forward(self, x:torch.Tensor)->torch.Tensor:
    # get the output from last hidden state that already pooled -> shape: [1, 1024]
    pooled_output = self.convnext(x).pooler_output
    # pass to classifier in order to get logits
    logits = self.classifier(pooled_output)
    return logits
  


