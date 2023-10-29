from src.production.models.convnext.core import ConvNextTransfer
from src.production.models.convolutional_autoencoder.core import (
    ConvolutionalAutoencoder,
    CAEncoderClassifier
)
from src.production.models.classifier import load_classifier_model, init_model
import torch

print(torch.__version__)
# extractor, convnext = load_classifier_model()
# test_model = ConvNextTransfer(convnext.convnext)


# test_model.load_state_dict(model_weight)
cae = ConvolutionalAutoencoder()
classifier = CAEncoderClassifier(8, 133, cae.encoder)

save_path = '/workspaces/dog-breed-classification-webapp/src/production/models/model_weights/'
torch.save(classifier.state_dict(), f'{save_path}/test_weight.pth')

classifier.load_state_dict(torch.load(f'{save_path}/test_weight.pth'))
print("weight loaded success!")
