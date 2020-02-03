#Import Packages and Functions
print('\033[1mImporting More Packages Than Necessary\033[0m')
import os
import sys
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tifffile import imsave
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Dropout
import glob
K.set_image_dim_ordering('tf')
import tensorflow as tf
import math
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import staintools
import cv2 as cv
from staintools.preprocessing.input_validation import is_uint8_image
import logging
import warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')
logging.getLogger('tensorflow').disabled = True
def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return ( 2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
#End Import Packages


#Editable Parameters
SliceLength=5000

file_location='./Inputs/*.tif'
save_location='./Outputs/'
NeoplasiaThreshold=.7  #Threshold optimized on prior data
MetaplasiaThreshold=.5 #Threshold optimized on prior data
NormalThreshold=.3     #Threshold optimized on prior data

#Non-editable Parameters
SquareTileLength=512 
HalfLength=256
IMG_CHANNELS = 3


#Load UNet Models
print('\033[1mLoading My Awesome Models\033[0m')
ModelSaveNameADM='MicePan-ADM-512-2Tone-T2-10Normal-2Ductal-Ep50-B32-L7E4'
ModelSaveNameDuc='MicePan-Ductal-512-2Tone-T2-5ADM-5Normal-Ep50-B32-L7E4'
ModelSaveNameNorm='MicePan-Normal-512-2Tone-T2-10ADM-2Ductal-Ep50-B32-L7E4'
modelADM= load_model(ModelSaveNameADM+'.h5', custom_objects={'dice_coef': dice_coef})
modelDuc= load_model(ModelSaveNameDuc+'.h5', custom_objects={'dice_coef': dice_coef})
modelNorm= load_model(ModelSaveNameNorm+'.h5', custom_objects={'dice_coef': dice_coef})

#Initialize Normalizer
class ReinhardColorNormalizer(object):
    """
    Normalize a patch color to the target image using the method of:
    E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley,
    'Color transfer between images'
    """
    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target,mask):
        """
        Fit to a target image
        :param target: Image RGB uint8.
        :return:
        """
        means, stds = self.get_mean_std(target,mask)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I,mask):
        """
        Transform an image.
        :param I: Image RGB uint8.
        :return:
        """
        I1, I2, I3 = self.lab_split(I)
        means, stds = self.get_mean_std(I,mask)
        norm1 = ((I1 - means[0]) * ((self.target_stds[0] +.000001)/ (stds[0]+.000001))) + self.target_means[0]
        norm2 = ((I2 - means[1]) * ((self.target_stds[1] +.000001)/ (stds[1]+.000001))) + self.target_means[1]
        norm3 = ((I3 - means[2]) * ((self.target_stds[2] +.000001)/ (stds[2]+.000001))) + self.target_means[2]

        return self.merge_back(norm1, norm2, norm3)
    @staticmethod
    def lab_split(I):
        """
        Convert from RGB uint8 to LAB and split into channels.
        :param I: Image RGB uint8.
        :return:
        """
        assert is_uint8_image(I), "Should be a RGB uint8 image"
        I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        I_float = I.astype(np.float32)
        I1, I2, I3 = cv.split(I_float)
        I1 /= 2.55  # should now be in range [0,100]
        I2 -= 128.0  # should now be in range [-127,127]
        I3 -= 128.0  # should now be in range [-127,127]
        return I1, I2, I3
    @staticmethod
    def merge_back(I1, I2, I3):
        """
        Take seperate LAB channels and merge back to give RGB uint8.
        :param I1: L
        :param I2: A
        :param I3: B
        :return: Image RGB uint8.
        """
        I1 *= 2.55  # should now be in range [0,255]
        I2 += 128.0  # should now be in range [0,255]
        I3 += 128.0  # should now be in range [0,255]
        I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
        return cv.cvtColor(I, cv.COLOR_LAB2RGB)
    def get_mean_std(self, I,mask):
        """
        Get mean and standard deviation of each channel.
        :param I: Image RGB uint8.
        :return:
        """
        assert is_uint8_image(I), "Should be a RGB uint8 image"
        I1, I2, I3 = self.lab_split(I)
        m1=np.mean(I1[mask])
        sd1=np.std(I1[mask])
        m2=np.mean(I2[mask])
        sd2=np.std(I2[mask])
        m3=np.mean(I3[mask])
        sd3=np.std(I3[mask])
        means = m1, m2, m3
        stds = sd1, sd2, sd3,
        return means, stds
#End Normalizer


#Collecting Files to run
print('\033[1mCollecting Files To Analyze\033[0m')
test_ids=sorted(glob.glob(file_location))
print('\033[1mCollected '+str(len(test_ids))+' file(s)\033[0m')
#End File Collection


#Fit Normalizer
print('\033[1mFitting The Normalizer\033[0m')
target = imread("./TargetForNormalization-Copy1.tif")
target = staintools.LuminosityStandardizer.standardize(target)
normalizer = ReinhardColorNormalizer()
mask1=target[:,:,0]<=200
mask2=target[:,:,1]<=200
mask3=target[:,:,2]<=200
mask=(mask1+mask2+mask3)
normalizer.fit(target,mask)
#End Fit Normalizer


#Begin Analysis Pipeline
print('\033[1mBeginning Analysis\033[0m')
sys.stdout.flush()
sizes_test = []
X_test = np.zeros((2,SquareTileLength, SquareTileLength,3), dtype=np.uint8)
p=0

for file in test_ids: 
    #Loop through all image files
    
    p+=1
    FullImage=np.asarray(imread(file))
    name=file.split('/')[-1]
    col, row,ch= FullImage.shape
    MeanTimesMat=np.zeros((col,row))
    NewImgADM= np.zeros((col,row))
    NewImgDuc= np.zeros((col,row))
    NewImgNorm =np.zeros((col,row))
    l=len(range(SliceLength,row-(SliceLength-1),int(SliceLength)))
    
    # setup progressbar
    print('\033[1mRunning Image: ' +str(p)+'/'+str(len(test_ids))+'\033[0m' )
    sys.stdout.write("Progress: [%s]" % (" " * (20+1)))
    sys.stdout.flush()
    sys.stdout.write("\b" * (20+1)) 
    progress=0
                  
    for n,a in enumerate(list(range(SliceLength,row-(SliceLength-1),int(SliceLength)))):
        #Loop through rows of image making intermediate crops
        
        for b in range(SliceLength,col-(SliceLength-1),int(SliceLength)):
            #Loop through columns of image making intermediate crops
            
            #Read and Normalize Intermediate Crop
            image=FullImage[b-(SliceLength):b+(SliceLength),a-(SliceLength):a+(SliceLength)] 
            mask1=image[:,:,0]<=200
            mask2=image[:,:,1]<=200
            mask3=image[:,:,2]<=200
            mask=(mask1+mask2+mask3)
            to_transform = staintools.LuminosityStandardizer.standardize(image)
            transformed = normalizer.transform(to_transform,mask)
            transformed[~mask]=image[~mask]
            
            #Set up storage variables
            NormCol, NormRow,ch= transformed.shape
            MeanTimesSlice=np.zeros((NormCol,NormRow))
            NewSliceADM= np.zeros((NormCol,NormRow))
            NewSliceDuc= np.zeros((NormCol,NormRow))
            NewSliceNorm =np.zeros((NormCol,NormRow))
            
            for i in range(HalfLength,NormRow-(HalfLength-1),HalfLength):
                #Loop through rows of intermediate crops making tiles for analysis
                
                for j in range(HalfLength,NormCol-(HalfLength-1),HalfLength):
                    #Loop through columns of intermediate crops making tiles for analysis
                    
                    #Crop tile and predict tile
                    block=transformed[j-(HalfLength):j+(HalfLength),i-(HalfLength):i+(HalfLength)] 
                    if block.shape != (SquareTileLength, SquareTileLength,IMG_CHANNELS):
                        block=resize(block, (SquareTileLength, SquareTileLength,IMG_CHANNELS), mode='constant', preserve_range=True)
                    X_test[-1]=block
                    preds_test_ADM = modelADM.predict(X_test)
                    preds_test_Duc = modelDuc.predict(X_test)
                    preds_test_Norm = modelNorm.predict(X_test)
                    
                    #Add tile predictions back to intermediate crop mask
                    MeanTimesSlice[j-(HalfLength):j+(HalfLength),i-(HalfLength):i+(HalfLength)]+=1
                    NewSliceADM[j-(HalfLength):j+(HalfLength),i-(HalfLength):i+(HalfLength)]+=np.squeeze(preds_test_ADM[-1])
                    NewSliceDuc[j-(HalfLength):j+(HalfLength),i-(HalfLength):i+(HalfLength)]+=np.squeeze(preds_test_Duc[-1])
                    NewSliceNorm[j-(HalfLength):j+(HalfLength),i-(HalfLength):i+(HalfLength)]+=np.squeeze(preds_test_Norm[-1])

            #Add predicted intermediate crops back to full mask
            NewImgADM[b-(SliceLength):b+(SliceLength),a-(SliceLength):a+(SliceLength)] += NewSliceADM
            NewImgDuc[b-(SliceLength):b+(SliceLength),a-(SliceLength):a+(SliceLength)] += NewSliceDuc
            NewImgNorm[b-(SliceLength):b+(SliceLength),a-(SliceLength):a+(SliceLength)] += NewSliceNorm
            MeanTimesMat[b-(SliceLength):b+(SliceLength),a-(SliceLength):a+(SliceLength)] += MeanTimesSlice
        
            # update the progress bar
            if (n/l*100)>=(progress+5):
                sys.stdout.write("-")
                sys.stdout.flush()
                progress+=5
    
    sys.stdout.write("]  saving predictions...\n") # this ends the progress bar
    #delete old variables to make room in memory
    del(mask1)
    del(mask2)
    del(mask3)
    del(mask)
    del(to_transform)
    del(transformed)
    del(NewSliceADM)
    del(NewSliceDuc)
    del(NewSliceNorm)
        
    #Construct Predicted Masks
    AvgImgADM=(NewImgADM+.000001)/(MeanTimesMat+.000001)
    MetaplasiaMask=AvgImgADM>=.5 #Threshold variable chosen from prior optimization
    
    AvgImgDuc=(NewImgDuc+.000001)/(MeanTimesMat+.000001)
    NeoplasiaMask=AvgImgDuc>=.7 #Threshold variable chosen from prior optimization
    
    AvgImgNorm=(NewImgNorm+.000001)/(MeanTimesMat+.000001)
    NormMask=AvgImgNorm>=.3 #Threshold variable chosen from prior optimization
    
    #Delete old variable to make room in memory
    del(MeanTimesMat)
    del(AvgImgADM)
    del(AvgImgDuc)
    del(AvgImgNorm)
    
    #Construct stromal mask
    mask1=FullImage[:,:,0]>=200
    mask2=FullImage[:,:,1]>=200
    mask3=FullImage[:,:,2]>=200
    mask=mask1+mask2+mask3
    del(mask1)
    del(mask2)
    del(mask3)
    stromal=mask==False

    #Combine Masks
    NeoplasiaMask[MetaplasiaMask==1]=0
    NeoplasiaMask[NormMask==1]=0
    MetaplasiaMask[NormMask==1]=0
    
    #Remove white space predictions
    NeoplasiaMask[mask]=0
    MetaplasiaMask[mask]=0
    NormMask[mask]=0
    del(mask)
    
    #Combine Masks
    stromal[NeoplasiaMask==1]=False
    stromal[MetaplasiaMask==1]=False
    stromal[NormMask==1]=False
    
    
    #Create Ouput Image
    AdjustedImage=np.zeros(np.shape(FullImage))

    AdjustedImage[NeoplasiaMask==1,0]=230
    AdjustedImage[NeoplasiaMask==1,1]=210
    AdjustedImage[NeoplasiaMask==1,2]=30

    AdjustedImage[MetaplasiaMask==1,0]=222
    AdjustedImage[MetaplasiaMask==1,1]=31
    AdjustedImage[MetaplasiaMask==1,2]=123

    AdjustedImage[NormMask==1,0]=122
    AdjustedImage[NormMask==1,1]=230
    AdjustedImage[NormMask==1,2]=213
    
    AdjustedImage[stromal,0]=19
    AdjustedImage[stromal,1]=16
    AdjustedImage[stromal,2]=163
    
    #Save Prediction Masks
    imsave(save_location+'Predicted-Metaplasia-'+name,np.squeeze(MetaplasiaMask.astype('uint8')), compress=6,  bigtiff=True)
    imsave(save_location+'Predicted-Neoplasia-'+name,np.squeeze(NeoplasiaMask.astype('uint8')), compress=6,  bigtiff=True)
    imsave(save_location+'Predicted-Normal-'+name,np.squeeze(NormMask.astype('uint8')), compress=6,  bigtiff=True)
    imsave(save_location+'Predicted-Stromal-'+name,np.squeeze(stromal.astype('uint8')), compress=6,  bigtiff=True)
    imsave(save_location+'CombinedImage-'+name,np.squeeze(AdjustedImage.astype('uint8')), compress=6,  bigtiff=True)
    
    del(MetaplasiaMask)
    del(NeoplasiaMask)
    del(NormMask)
    del(stromal)
    del(AdjustedImage)
    print('\033[1mFinished Image: ' +str(p)+'/'+str(len(test_ids))+'\033[0m' )
            
print('Done')
#End Analysis Pipeline










