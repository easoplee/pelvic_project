# Pelvic Region 3D MRI Scan Bone Segmentation Website

## The Project Flowchart

![The project flow](figs/csplus_flowchart.png)

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
- Transformer (still in progress)

Set the model type ('unet', 'deeplabv3') and set the variable training to True to train the specific model.

3D models
- U-NET

## Results

2D binary and multi-class segmentation acheiving dice score of 0.7-0.9. Trained for 30 epochs. Check out the figs/boxplots directory for boxplot result for each model and bone segmentation. More documentation will be added under this section in the future.

## 2D to 3D Model

Under the CS+ Website folder, the python script called plotly_3d.py is used to create an interactable 3D model. For the 3D construction technique, marching cube algorithm from skimage package is used. It creates a volumetric isosurface from a z-stack 2D segmentation slices.

## Website

Go into the CS+ Website folder, run the web.py script in local terminal to launch the website. To view a 3D model of the selected bone type, go to the "Input Images" tab. Then, input a 3D file (.npy) along with the bone type. The loading time will be approximately 20-30 seconds before the app redirects to another page that contains an interactable bone model. 

The website currently supports an input of 3D pelvic MRI scan and is able to output an interactable 3D display of the bone the user chooses (femur, spine, sacrum, ilium).

## Future Plans

We plan to continue adding more features to the website. 