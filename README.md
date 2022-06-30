# Pelvic Region 3D MRI Scan Bone Segmentation

The objective of this project is to perform automatic multi-class bone segmentation on pelvic MRI dataset using deep learning models. Each pixel is assigned to one of the following classes.

Classes:

0. background
1. femur
2. ilium
3. spine
4. sacrum

The objectives of the binary segmentation projects are to perform automatic binary bone segmentation on pelvic MRI dataset. Each pixel is assigned to one of the two classes.

Classes:

0. background
1. bone

The binary segmentation project is done for each type of the bone visible in the pelvic MRI (femur, ilium, spine, sacrum). 

## Segmenation Mask

Finished working on creating manual segmentation mask for bone segmentation project. The final segmented mask is under the mask folder. The folder contains binary segmented masks and multiclass segmented masks.

## Model

2D models
- U-Net with ResNet-34 as encoder
- DeepLabV3
- Segmenter

Set the model type and set the variable training to True to run the specific model.

3D models
- U-NET

## Results

2D binary and multi-class segmentation acheiving dice score of 0.8-0.9. More documentation will be added under this section in the future.
