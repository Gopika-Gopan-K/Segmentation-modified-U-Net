import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from train_data_prep import train_data_new, train_label_new
from model import Unet
from torch.optim import Adam
import warnings
from torch.cuda import amp
from tqdm import tqdm
from loss import FocalTverskyLoss
import copy
import torchvision.transforms as T


# Ignore warnings
warnings.filterwarnings("ignore")

# Create GradScaler for mixed precision training
scaler = amp.GradScaler()

# Choose the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Function to crop mask to the size of output of U-net
def crop_feat(inp):
    inp_size = inp.size()[2]  # here it is 572 as decided during data preparation
    delta = (inp_size-388) // 2
    return inp[:, delta:inp_size-delta, delta:inp_size-delta]


# Function to augment data
def data_augmentation(image, mask):
    # Horizontal flip
    if torch.rand(1) > 0.5:
        image = F.hflip(image)
        mask = F.hflip(mask)
    # Vertical flip
    if torch.rand(1) > 0.5:
        image = F.vflip(image)
        mask = F.vflip(mask)
    # Rotate 90 degree or -90 degree
    if torch.rand(1) > 0.5:
        if torch.rand(1) > 0.5:
            image = torch.rot90(image, 1, [0, 1])
            mask = torch.rot90(mask, 1, [0, 1])
        else:
            image = torch.rot90(image, -1, [0, 1])
            mask = torch.rot90(mask, -1, [0, 1])

    if torch.rand(1) > 0.5:
        image=F.adjust_gamma(image, gamma=0.75)


    return image, mask


# Dataset class
class CovidSegData(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return self.labels.size()[2]  # Data in (H,W,C) format obtained from data preparation

    def __getitem__(self, item):
        # Select each slice
        slices = self.data[:, :, item]
        masks = self.labels[:, :, item]
        slices=F.autocontrast(slices.unsqueeze(0))
        if self.transform is not None:  # if transform is True carry out data augmentation
            slices, masks = data_augmentation(slices.squeeze(), masks)
        return slices, masks




# Loss function
def loss_function(preds1, preds2, preds3, preds4, GT):
    # resize groundtruth according to the predicted label dimensions
    GT1a = GT  # Ground truth already cropped to the output dimension of Unet
    resize_mask2 = T.Resize(size=(preds2.size()[2], preds2.size()[3]))
    GT2 = resize_mask2(GT)
    resize_mask3 = T.Resize(size=(preds3.size()[2], preds3.size()[3]))
    GT3 = resize_mask3(GT)
    resize_mask4 = T.Resize(size=(preds4.size()[2], preds4.size()[3]))
    GT4 = resize_mask4(GT)


    D1f = FocalTverskyLoss(preds1, GT1a, gamma=1)
    D2f = FocalTverskyLoss(preds2, GT2)
    D3f = FocalTverskyLoss(preds3, GT3)
    D4f = FocalTverskyLoss(preds4, GT4)
    # weighted loss -- deep supervision
    FTL_final = (0.5 * D1f)+(0.2 * D2f)+(0.2 * D3f)+(0.1 * D4f)

    loss_final = FTL_final

    return loss_final


# Load data and dataloaders for training and validation
train_dataset = CovidSegData(train_data_new, train_label_new, transform=True)
train_loader = DataLoader(train_dataset, batch_size=30,
                          num_workers=2, shuffle=True, pin_memory=True)


# Initialize network

model=Unet()
model.to(device)  # Move the model to GPU



# Optimizer and Scheduler
optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
num_epochs = 50  # number of epochs

best_epoch_loss = 1


for epoch in range(num_epochs):
    dataset_size = 0
    running_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)  # for progress bar display
    for batch_idx, (images, GT1) in loop:
        model.train()  # model in training mode

        # Crop mask to the size of output of Unet,i.e, 388 in this case
        GT = crop_feat(GT1)
        # Get data to cuda
        images = images.to(device)
        GT = GT.to(device)

        batch_size = images.size()[0]  # Size of each batch
        optimizer.zero_grad()  # zeroing gradients
        images = torch.unsqueeze(images, 1)  # get correct input dimensions
        images = images.type(torch.cuda.FloatTensor)  # Convert input to Float tensor
        with amp.autocast():  # forward part with autocasting -- mixed precision training (MPT)
            preds1, preds2, preds3, preds4 = model(images)  # predictions
            loss = loss_function(preds1, preds2, preds3, preds4, GT)  # loss
        scaler.scale(loss).backward()  # scales loss and create scaled gradients for MPT
        # unscale the gradients of the optimizer assigned params, skips optimizer.step if Nan or Inf present
        scaler.step(optimizer)
        scaler.update()  # update scale for next iteration

        # Epoch loss calculation
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size



        # Update progress bar
        loop.set_description(f"Epoch : [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item(), epoch_loss=epoch_loss)

    if epoch_loss < best_epoch_loss:
        print(f"Loss Improved ({best_epoch_loss} ---> {epoch_loss})")
        best_epoch_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        PATH = "Loss{:.4f}_epoch{:.0f}_final.bin".format(best_epoch_loss, epoch)
        torch.save(model.state_dict(), PATH)
        print("Model Saved")
torch.save(model, f'model_final.pth')
