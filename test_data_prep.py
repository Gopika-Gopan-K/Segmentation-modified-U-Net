import nibabel as nib
import os
import torch


# Data and mask path
data_path = ".../data/"
mask_path = "...masks/"

# Lung window in HU units
HU_min = -1000
HU_max = 400

# names of files in data folder
arr = os.listdir(data_path)
print(len(arr))

# select val and val subjects
test_data_numbers = range(0, 50)


# function to preprocess data
def data_preprocess(path, hu_min, hu_max):
    volume_data = nib.load(path)  # load data
    volume_data_numpy = volume_data.get_fdata()  # get data as numpy
    volume_data_tensor = torch.tensor(volume_data_numpy)  # convert to torch tensor
    volume_data_tensor_clamped = torch.clamp(volume_data_tensor, min=hu_min, max=hu_max)  # apply HU lung window
    volume_data_tensor_clamped_normalized = (volume_data_tensor_clamped-hu_min) / (hu_max-hu_min)  # normalize to [0,1]
    return volume_data_tensor_clamped_normalized


# function to obtain maask
def mask_obtain(fpath):
    mask = nib.load(fpath)  # load mask
    mask_numpy = mask.get_fdata()  # get mask as numpy
    mask_tensor = torch.tensor(mask_numpy)  # convert to torch tensor
    return mask_tensor


# add zero padding if size of image less than required input size
def padding_size(slices):
    if (572-slices.size()[1]) % 2 == 0:  # See if the difference between required size and data size is even or odd
        # if difference is even, pad same number to either side
        pad1 = (572-slices.size()[1]) // 2
        pad2 = (572-slices.size()[1]) // 2
    else:
        # if difference is even, pad one side one value more than other
        pad1 = (572-slices.size()[1]) // 2
        pad2 = ((572-slices.size()[1]) // 2)+1

    if (572-slices.size()[2]) % 2 == 0:  # See if the difference between required size and data size is even or odd
        # if difference is even, pad same number to either side
        pad3 = (572-slices.size()[2]) // 2
        pad4 = (572-slices.size()[2]) // 2
    else:
        # if difference is even, pad one side one value more than other
        pad3 = (572-slices.size()[2]) // 2
        pad4 = ((572-slices.size()[2]) // 2)+1

    return [pad4, pad3, pad2, pad1]  # return the number of zero padding in each side of the slice


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# test data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialize for stacking
test_data = torch.empty((1, 572, 572))
test_label = torch.empty((1, 572, 572))

# Function to form val data and val label
for i in test_data_numbers:
    print(i)
    file_path = data_path+arr[i]  # path of the data
    data = data_preprocess(file_path, HU_min, HU_max)  # preprocess data
    data = data.permute(2, 0,
                        1)  # change the dimension (H,W,C) ---> (C,H,W) , since ConstantPad2d works with this config
    P = padding_size(data)  # Obtain the required padding sizes
    data = torch.nn.ConstantPad2d(P, 0)(data)  # pad the slices according to the padding sizes obtained
    test_data = torch.cat((test_data, data), 0)  # stack the data(slices) along dimension C

    file_path_mask = mask_path+arr[i]  # path to the mask
    label = mask_obtain(file_path_mask)  # obtain the mask
    # NOTE: Since we padded the data, mask should also have same size, so pad mask also
    label = label.permute(2, 0,
                          1)  # change the dimension (H,W,C) ---> (C,H,W) , since ConstantPad2d works with this config
    label = torch.nn.ConstantPad2d(P, 0)(label)  # pad the maks according to the padding sizes of the slices
    test_label = torch.cat((test_label, label), 0)  # stack the masks along dimension C

# remove the empty
test_data = test_data[1:test_data.size()[0], :, :]
test_label = test_label[1:test_label.size()[0], :, :]
# Determine which slices are all black
idx = []
for i in range(test_label.size()[0]):
    img_max = torch.max(test_label[i, :, :])
    if img_max == 1:
        idx.append(i)  # having white regions

# Choose data without completely black mask, i.e, having atleast some white segmented region
test_data_new1 = test_data[idx, :, :]
test_label_new1= test_label[idx, :, :]
# (C,H,W) ---> (H,W,C) since Dataset class has this config (this part is not necessary if we change the config of
# Dataset class)
test_data_new = test_data_new1.permute(1, 2, 0)
test_label_new = test_label_new1.permute(1, 2, 0)


