#Evaluate top one accuracy prediction prediction of the net using the coco evaluation data set

# 1. Train net or Download pre trained net weight from [here](https://drive.google.com/file/d/1xRFvBk_PONwJHmP2NcwFaEQc_Z_JInpE/view?usp=sharing).
# 2. If not already set, download and set coco API and data set (See instruction)
# 3. Set Train coco image folder path  in: TestImageDir
# 4. Set the path to the coco Train annotation json file in: TestAnnotationFile
# 5. Run the script

#------------------------------------------------------------------------------------------------------------------------
# Getting COCO dataset and API
# Download and extract the [COCO 2014 train images and Train/Val annotations](http://cocodataset.org/#download)
# Download and make the COCO python API base on the instructions in (https://github.com/cocodataset/cocoapi).
# Copy the pycocotools from cocodataset/cocoapi to the code folder (replace the existing pycocotools folder in the code).
# Note that the code folder already contain pycocotools folder with a compiled API that may or may not work as is.
##########################################################################################################################################################################

import numpy as np
import Resnet50Attention as Net
import COCOReader as COCOReader
import os
import scipy.misc as misc
import torch
import numpy as np

#...........................................Input Parameters.................................................

Trained_model_path="logs/ModelWeightCOCOMaskRegionClassification.torch" # Weights for Pretrained net
TestImageDir='/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/val2014/' #COCO evaluation image dir
TestAnnotationFile = '/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/annotations/instances_val2014.json' # COCO Val annotation file
SamplePerClass=10
UseCuda=True

#---------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=COCOReader.COCOReader(TestImageDir,TestAnnotationFile)
NumClasses = Reader.NumCats

#---------------------Initiate neural net------------------------------------------------------------------------------------
Net=Net.Net(NumClasses=NumClasses,UseGPU=UseCuda)
Net.AddAttententionLayer() #Load attention layers

Net.load_state_dict(torch.load(Trained_model_path)) #load weights
if UseCuda: Net.cuda()
Net.eval()
#--------------------Evaluate net accuracy---------------------------------------------------------------------------------
CorCatPred=np.zeros([Reader.NumCats],dtype=np.float64) # Counter of correct class prediction

for c in range(Reader.NumCats): # Go over all classes and calculate accuracy
    #  print("Class "+str(c)+") "+Reader.CatNames[c])
      for m in range(np.min((SamplePerClass,len(Reader.ImgIds)))): # Go over images
            Images,SegmentMask,Labels, LabelsOneHot=Reader.ReadSingleImageAndClass(ClassNum=c,ImgNum=m) #Load Data
            Prob, PredLb = Net.forward(Images, ROI=SegmentMask,EvalMode=True)  # Run net inference and get prediction
            PredLb = np.array(PredLb.data)
            Prob = np.array(Prob.data)
            if PredLb[0]==Labels[0]: CorCatPred[c]+=1 # Check if prediction is correct
            # print("Real Label " +Reader.CatNames[Labels[0]]+" Predcicted Label "+Reader.CatNames[PredLb[0]])
            # print("Predicted Label Prob="+str(Prob[0,PredLb[0]])+  " Real Label predicted prob="+str(Prob[0,Labels[0]]))
            # Images[:, :, :, 0] *= SegmentMask
            # misc.imshow(Images[0])

      CorCatPred[c]/=np.min((SamplePerClass,len(Reader.ImgIds))) #Calculate class prediction accuracy
      print("Class " + str(c) + ") " + Reader.CatNames[c] + "\t" +str(CorCatPred[c]*100)+"%")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Mean Classes accuracy \t" + str(CorCatPred.mean()*100)+"%") # Calculate mean accuracy for all classes
print("Number of samples per class\t"+str(SamplePerClass))


#--------------------------- Create files for saving loss----------------------------------------------------------------------------------------------------------

