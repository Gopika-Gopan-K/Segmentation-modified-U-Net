import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from test_data_prep import test_data_new, test_label_new
import warnings
from tqdm import tqdm
from loss import DCE

# Ignore warnings
warnings.filterwarnings("ignore")

# Choose the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to crop mask to the size of output of U-net
def crop_feat(inp):
    inp_size = inp.size()[2]  # here it is 572 as decided during data preparation
    delta = (inp_size-388) // 2
    return inp[:, delta:inp_size-delta, delta:inp_size-delta]

# Dataset class
class CovidSegData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.labels.size()[2]  # Data in (H,W,C) format obtained from data preparation

    def __getitem__(self, item):
        # Select each slice
        slices = self.data[:, :, item]
        masks = self.labels[:, :, item]
        slices = F.autocontrast(slices.unsqueeze(0))
        slices=slices.squeeze()
        return slices, masks


# Loss function
def loss_function(preds1, GT):
    # resize groundtruth according to the predicted label dimensions
    GT1a = GT  # Ground truth already cropped to the output dimension of Unet
    # Dice score for each predictions
    D1 = DCE(preds1, GT1a)
    loss_final = D1
    return 1-loss_final


# Load data and dataloaders

test_dataset = CovidSegData(test_data_new, test_label_new)
test_loader = DataLoader(test_dataset, batch_size=1,
                        num_workers=1, shuffle=False, pin_memory=True)
print('dataloaded')

model=torch.load('model_final.pth')
model.to(device)

print('Start ')
Dice_score=torch.empty(len(test_loader))
i=0
model.train()  # Set model in evaluation mode
with torch.no_grad():  # Disable gradient calculation
    dataset_size = 0
    running_loss = 0.0
    loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)  # For progress bar
    for batch_idx, (images, GT1) in loop:

        # Crop mask to the size of output of Unet,i.e, 388 in this case
        GT = crop_feat(GT1)
        # Get data to cuda
        images = images.to(device)
        GT = GT.to(device)

        batch_size = images.size()[0]  # Size of each batch
        images = torch.unsqueeze(images, 1)  # get correct input dimensions
        images = images.type(torch.cuda.FloatTensor)  # Convert input to Float tensor

        preds1,preds2,preds3,preds4 = model(images)  # predictions
        # preds1= model(images)  # predictions
        Dice = loss_function(preds1,  GT)  # loss
        Dice_score[i]=Dice.item()
        i+=1
        # Epoch loss calculation
        running_loss += (Dice.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        # Update progress bar
        loop.set_postfix(Dice=Dice.item(), epoch_loss=epoch_loss)

print(torch.mean(Dice_score))



















