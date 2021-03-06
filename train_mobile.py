import torch
from torchvision import transforms
from model import MobileNetv2
from dataloader import CustomDataset
from torch.optim import Adam
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import tqdm as tq
import numpy as np
from collections import Counter

CP_DIR = ""
LAST_CPT_DIR = ""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# update the path manually
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
NUM_CLASSES = 8
CONTINUE_EPOCH = 0

train_dataset = CustomDataset("train")
val_dataset = CustomDataset("val")

trainLoader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
valLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=8)

model = MobileNetv2(NUM_CLASSES).to(device)
# model.load_state_dict(torch.load(LAST_CPT_DIR))
optimizer = Adam(model.parameters(), lr=LR,
                 weight_decay=5e-4, amsgrad=True)
loss = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
print("Finished loading model, initiating session...")
errors_train = []
errors_valid = []
accu_valid = []
best_val_loss = 100.0
best_val_accu = 0.0
for epoch in range(EPOCHS):
    running_train_loss = 0.0
    running_valid_loss = 0.0
    running_valid_acc = 0.0

    for (input, label) in tq.tqdm(trainLoader):
        model.train()
        optimizer.zero_grad()
        input = input.to(device)
        label = label.to(device)

        output = model(input)

        error = loss(output, label)
        error.backward()
        optimizer.step()

        running_train_loss += error.item()
    epoch_train_loss = running_train_loss / len(trainLoader)
    errors_train.append(epoch_train_loss)
    print('Trained, Epoch {} - Loss {}'.format(CONTINUE_EPOCH +
                                               epoch+1, epoch_train_loss))
    print("")

    model.eval()
    for (input, label) in tq.tqdm(valLoader):
        input = input.to(device)
        label2 = label.clone()
        label2 = label2.cpu().numpy()
        label = label.to(device)

        with torch.no_grad():
            output = model(input)
        pred_label = torch.argmax(output, dim=1)
        pred_label = pred_label.cpu().numpy()

        accuracy = np.count_nonzero(pred_label == label2) / BATCH_SIZE

        # accuracy
        error = loss(output, label)
        running_valid_loss += error.item()
        running_valid_acc += accuracy
    epoch_valid_loss = running_valid_loss / len(valLoader)
    epoch_valid_acc = running_valid_acc / len(valLoader)
    IMPROVED = False
    if epoch_valid_acc > best_val_accu or epoch == 0:
        best_val_accu = epoch_valid_acc
    if epoch_valid_loss < best_val_loss or epoch == 0:
        best_val_loss = epoch_valid_loss
        IMPROVED = True
    errors_valid.append(epoch_valid_loss)
    accu_valid.append(epoch_valid_acc)
    scheduler.step()
    print('Validated, Epoch {} - Loss {} - Acc {}'.format(CONTINUE_EPOCH +
                                                          epoch+1, epoch_valid_loss, epoch_valid_acc))
    print("")
    if IMPROVED == True or epoch == 0:
        torch.save(model.state_dict(), os.path.join(
            CP_DIR, "MobileNet_epoch{}_valLoss{}_valAcc{}.pth".format(CONTINUE_EPOCH+epoch+1, epoch_valid_loss, epoch_valid_acc)))
        print("Saved checkpoint at epoch {} with validLoss= {} and validAccu= {}".format(
            CONTINUE_EPOCH+epoch+1, epoch_valid_loss, epoch_valid_acc))
    elif epoch_valid_loss >= best_val_loss:
        print("Valid loss {} did not improve from {}, not saving checkpoint, continuing...".format(
            epoch_valid_loss, best_val_loss))
