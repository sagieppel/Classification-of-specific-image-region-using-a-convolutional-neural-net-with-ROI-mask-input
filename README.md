# Classification of specific image region using a convolutional neural net with ROI mask input

Given an image and region of interest (ROI) mask as input, the net classifies the region of the image marked in the mask. 

![](/Figure1.png)
Figure 1. Segment-specific Classification using CNN

For example, in Figure 1  the net is given the same image twice with different masks and output the object class of the segment marked by each mask.

This net achive 83% accuracy on the coco objects data set.

For more details see [Classifying a specific image region using convolutional nets with an ROI mask as input](https://arxiv.org/pdf/1812.00291.pdf)


[## *Stronger but harder to trai/use version of this net can be found here](https://github.com/sagieppel/Generator-evaluator-selector-net-a-modular-approach-for-panoptic-segmentation/tree/master/Classification)


![](/Figure2.png)
Figure 2.a Standard Classification net, b. Region specific classification net

## Architecture and attention mechanism.
The net architecture can be seen in Figure 2.b. The net main branch consists of standard image classification neural net (Resnet 50). 
The side branch responsible for focusing the attention of the classifier net on the input mask region, in order to achieve region specific classification of the mask segment.
As shown in Figure 2.b attention focusing is done by modifying the features map of the image classification branch in several different layers according to the following steps:
1. For each layer, the ROI mask is resized to the size of the features map produce by that layer.
2. For each resized ROI mask two convolution layers (+bias) are applied to generate two attentions maps in the same size and depth of the corresponding feature map.
   The first of this attention maps is multiplied element-wise with the feature map while the second map is added element-wise to resulting maps.
3. The result normalized feature map is used as an input for the next layer of the classification net.
### Net Weight initiation
For training: the weights of the image classification were initiated using Resnet 50 weight pre-trained on the Imagenet.  
The weights of filters used to generate the attention maps that were multiplied by the feature maps were initiate to zero while the bias of this filters was initiated to 1. 
The bias and filter weights of layers used to generate the map that was added to feature map were all initiated to zero. 
Hence, at training initation, the attention branch have zero effect on the net.    

# Using the net.
## Setup
This network was run with [Python 3.6 Anaconda](https://www.anaconda.com/download/) package and [Pytorch 0.3](https://pytorch.org/). 

## For training and evaluating download COCO dataset and API

1. Download and extract the [COCO 2014 train images and Train/Val annotations](http://cocodataset.org/#download)
2. Download and make the COCO python API base on the instructions in (https://github.com/cocodataset/cocoapi). Copy the pycocotools from cocodataset/cocoapi to the code folder (replace the existing pycocotools folder in the code). Note that the code folder already contain pycocotools folder with a compiled API that may or may not work as is. 

## Prediction/Inference

1. Train net or download pre trained net weight from [here](https://drive.google.com/file/d/1xRFvBk_PONwJHmP2NcwFaEQc_Z_JInpE/view?usp=sharing).
2. Open RunPrediction.py 
3. Set Path to the trained net weights  file in: Trained_model_path 
4. Run script to get prediction on the test set
4. Paths of input image and input ROI mask is given in ROIMaskFile and InputMaskFile parameters.
5. Test image and ROI maps are supplied in the Test folder

## Training

1. Open Train.py
2. Set Train image folder path  in: TrainImageDir 
3. Set the path to the coco Train annotation json file in: TrainAnnotationFile
4. Run the script
5. The trained net weight will appear in the folder defined in: logs_dir 
6. For other training parameters see Input section in train.py script

## Evaluating 

1. Train net or Download pre trained net weight from [here](https://drive.google.com/file/d/1xRFvBk_PONwJHmP2NcwFaEQc_Z_JInpE/view?usp=sharing).
2. Open EvaluateAccuracy.py
3. Set Train coco image folder path  in: TestImageDir
4. Set the path to the coco Train annotation json file in: TestAnnotationFile
5. Run the script



