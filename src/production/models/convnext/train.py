import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import ConvNextDogDataset
from core import ConvNextTransfer
from utils import init_model
from tqdm.auto import tqdm
from torchmetrics import Accuracy
import argparse

# init the CONSTANT variable
EPOCHS = 1
BATCH_SIZE = 4


# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-data_path", "--data_dir", required=True,
   help="Path of Training dataset")

args = vars(ap.parse_args())

# load the model from huggingface hub
extractor, convnext = init_model()
label2idx, idx2label = convnext.config.label2id, convnext.config.id2label


# this script is created to fine-tuned a ConvNext Model located on hf hub
# prepare a dataobject before pass it to dataloader
# this part can be improved by making it can take an argument for the script
data_dir = args['data_dir']


train_data = ConvNextDogDataset(
    convnext,
    extractor,
    data_dir, 
    fold='train'
)

validation_data = ConvNextDogDataset(
    convnext,
    extractor,
    data_dir, 
    fold='valid'
)

test_data = ConvNextDogDataset(
    convnext,
    extractor,
    data_dir, 
    fold='test'
)


train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataloader = DataLoader(
    dataset=validation_data,
    batch_size=BATCH_SIZE,
    shuffle=False # we are not gonna shuffle the test data
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # we are not gonna shuffle the test data
)

# load the model
transfer_model = ConvNextTransfer(
    convnext_block=convnext.convnext,
    num_labels=len(label2idx),
)


# prepare component
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transfer_model = transfer_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    params=transfer_model.parameters(),
    lr=1e-5
)
accuracy = Accuracy(task="multiclass", num_classes=len(label2idx)).to(device)


# to train a model
# write a training loop 
# we will keep loss of the best model

epochs = EPOCHS

best_loss = float('inf') # the lower the better
best_acc = float('-inf') # the higher the better
best_epoch = ""

# keep track of accuracy and loss of both train and val
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch+1}")
  train_loss, train_acc = 0, 0
  transfer_model.train()
  # loop through data batches
  try:
    for batch, (train_images, train_labels) in enumerate(train_dataloader):
      train_images, train_labels = train_images.to(device), train_labels.to(device)
      # forward pass
      logits = transfer_model(train_images)     
      # calculate loss
      loss = loss_fn(logits, train_labels)
      acc = accuracy(torch.softmax(logits, dim=1).argmax(dim=1), train_labels)
      train_loss += loss
      train_acc += acc
      # optimizer zero grad
      optimizer.zero_grad()
      # loss backward (backpropagation)
      loss.backward()

      # optimizer step (grad desc)
      optimizer.step()
    # finding an average loss and accuray of each epoch
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)
    # save the loss and acc of each epoch
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
  except OSError:
    pass

  # testing loop
  transfer_model.eval()
  val_loss, val_acc = 0, 0
  with torch.inference_mode():
    for val_images, val_labels in val_dataloader:
      # put data onto device 
      val_images, val_labels = val_images.to(device), val_labels.to(device)
      val_logits = transfer_model(val_images)
      val_loss += loss_fn(val_logits, val_labels)
      val_acc += accuracy(torch.softmax(val_logits, dim=1).argmax(dim=1), val_labels)
  
  val_loss = val_loss / len(val_dataloader)
  val_acc = val_acc / len(val_dataloader)
  val_loss_list.append(val_loss)
  val_acc_list.append(val_acc)

  # keep report when the best model appeared
  print('----------')
  if (val_loss < best_loss) and (val_acc > best_acc):
    best_loss = val_loss
    best_acc = val_acc
    best_epoch = str(epoch+1)
    # save the model weight
    torch.save(transfer_model.state_dict(), '/content/model_weights/best_model_convnext.pth')
    print(f"Best Model Saved!")

  print(f'train_loss: {train_loss:.2f} | train_acc: {train_acc:.2f} | val_loss {val_loss:.2f} | val_acc {val_acc:.2f}')
  print('----------')

print("Training Result:")
print(f"Best Model is at epoch {best_epoch} -> val_loss: {best_loss:.2f} | val_acc: {best_acc:.2f}")
# save the model weight
torch.save(transfer_model.state_dict(), '/content/model_weights/most_epoch_convnext.pth')
