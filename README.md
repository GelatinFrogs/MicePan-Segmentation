# Segmentation of Developing Cancer Morphologies in Mouse Pancreas
This is an image analysis pipeline that takes input H&E images of a mouse pancreas and determines the location and abundance of tissue types common in developing pancreatic cancer. The predicted tissues can both replicate immunostaining techniques, and can even discern morphologies that are inseperable with current stains.

Insert Diagram Here

## Method
This pipeline uses a stack of already trained UNet models [1] to predict tissue type based on morphology. Prior to UNet analysis, the H&E Images are locally normalized at intermediate crops using the Reinhard method [2]. Predictions for neoplasias, metaplasias, normal acinars, and stromal tissues are then combined to produce comprehensive results for an entire tissue.

Insert Diagram Here

## Prerequisites
- Linux or macOS
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN
- Tested on Python 3.7.3

## Necessary Packages
- keras
- tensorflow
- staintools v. 2.1.2
- numpy
- PIL
- OpenCV
- skimage
- tiffile

## Deploying the Pipeline
- Copy H&E Images of mouse pancereas with developing cancer into the "Inputs" folder.
- Run the Pipeline
```bash
python ProcessImages.py
```
- Predicted tissue masks and combined image be saved into the "Outputs" Folder

## Acknowledgements
1: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. ArXiv, abs/1505.04597. 
2: Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001). Color Transfer between Images. IEEE Computer Graphics and Applications, 21, 34-41





