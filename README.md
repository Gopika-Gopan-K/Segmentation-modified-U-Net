# Segmentation-modified-U-Net
The  2D segmentation framework for segmentation of lesions in CT slices is shown in figure.
![alt text](https://github.com/Gopika-Gopan-K/Segmentation-modified-U-Net/blob/d3c0637857f4fe6bd6d3665c1a5ad7e6af1a726d/pics/block_diagram.png)

The base framework is that of U-net. U-net has been modified by adding Attention module in each layer of the decoder. This is carried out to ensure that the model learns spatial and channel inter-dependencies effectively. Atrous Spatial Pyramid Pooling with residual connection has been added at the bottleneck layer as well as before the final output layer to enable the network to obtain the semantic information at multiple scales using different rates of Atrous convolution. In addition to these, the output at each level of decoder network is compared with the Ground Truth to ensure that the network focuses on the relevant features and any error occurring at the initial layers of decoder is not propagated to final output layer. 

The input are CT slices which are covid positive and the model outputs the lesion segmentation masks. 

Ensure the CT volumes are in folder named “Data” and lesion mask in folder “Lesion Mask”. The lesion mask should have same name as the corresponding CT volume and both should be in .nii format. The CT scans are extracted and lung window of -1000 HU to 400 HU is applied before the slices are normalized for further analysis.

