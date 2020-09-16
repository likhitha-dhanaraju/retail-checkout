# Retail Checkout Using Object Detection

### Overview : 
- We use an AI-based approach for retail billing, significantly reducing retail checkout times. Our model accurately detects multiple objects simultaneously, instantly generating bills at the click of a button. Customer satisfaction is vastly improved, leading to better sales, as they no longer need to wait for each item to be manually scanned before the bill is generated.
- Our product uses an object detection model to detect items from a single image. A list of the detected items is then passed to the Point of Sale hardware for bill generation.
- Google Colab notebooks are used for training and testing the object detector. These can be run on any computer with an internet connection.
- Training and test data are derived from the RPC dataset. It contains retail checkout images, with 5-20 items per image. These items belong to a set of 200 different product categories.    
  - Training, test data, and the object detection model can be found in the following drive folder : <br/>
		 [Drive Link](https://drive.google.com/open?id=10L1PLtX46ANJJ_RafmIREqSmqkeRz6_4). Please add the above folder to your Drive in order to train or test the object detector. 

### Training :
   1. Training images are found in the folder **traindata2019**.
   2. The training annotations, in COCO format, are contained in the file **train.json**.
   3. Follow the instructions in the colab notebook, **Smart Billing_Train.ipynb** to train the object detector.

### Testing :
   1. Test images are found in the folder **testdata2019**.
   2. The test annotations, in COCO format, are contained in the file **test.json**.
   3. The trained object detector is contained in the folder smart_billing_model. This can be used for testing.
   4. Follow the instructions in the colab notebook, **Smart_Billing_Test.ipynb** to test the object detector. The detected items can be passed as a list to the Point of Sale hardware for bill generation. 

### Deployment in AWS

  1. Test images are found in the folder **testdata2019**.
  2. The test annotations, in COCO format, are contained in the file **test.json**.
  3. Create a notebook instance, upload the **testdata2019**, **test.json**, **AWS/code/** folder  and follow the instructions in **AWS/Smart_Billing_Test.ipynb** to test the model in the notebook, and deploy it to the cloud.

#### Using custom data :
   1. Training and testing images are placed in the folders traindata2019 and testdata2019 respectively.
   2. Training and test annotations are prepared in the standard COCO format for Object Detection. A free, open-source tool for this can be found in the following link: https://github.com/jsbroks/coco-annotator
   3. The generated JSON files for training and test data are then uploaded as train.json and test.json respectively.
