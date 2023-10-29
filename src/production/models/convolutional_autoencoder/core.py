from torch import nn
import torch

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.Sigmoid()
        )
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# this is a classifier utilizing CVAE encoder component
class CAEncoderClassifier(nn.Module):
  '''An integration of Model with trained encoder from CVAE'''
  def __init__(self, 
               hidden_units: int, 
               output_shape: int,
               encoder: nn.Sequential
               ):
    
    super().__init__()
    self.cvae_encoder = encoder # this is for trained encoder (cvae)
    self.classifier = nn.Sequential(
        nn.Flatten(), # reshape it into shape of vectors
        nn.Linear(in_features=hidden_units*56*56, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=output_shape)
    )

  def forward(self, x:torch.Tensor)->torch.Tensor:
    compressed = self.cvae_encoder(x)
    logits = self.classifier(compressed)
    return logits