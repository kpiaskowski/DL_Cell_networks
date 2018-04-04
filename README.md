### Description
This is a TensorFlow implementation of simple yet powerful object detector, 
based on autoencoder architecture. It is fully convolutional which makes it really fast.

### How does it work?
The network takes the images of constant size w = h = 448 (it is not crucial to
resize images to the same size, nevertheless it was empirically tested as yielding best 
results). The network is trained to convert input images into
S × S × C tensor of probabilities, where S stands for the size of output tensor and C is the number of classes in the dataset.
Each cell in this tensor denotes probability that there is a part of some object within this cell.
The first two coordinates denotes spatial probability and ensures translation invariance, whereas
third parameter corresponds to class of object. 

Simple example: assuming S = 14 and C = 80 (as in COCO dataset) and the original object was 
of class no. 53 and occupies entire upper left part of image, then the network should produce tensor T,
where T[0:7, 0:7, 52] (52 because zero indexing, you know) is filled with ones. When there are other 
classes as well as overlapping object, there should be naturally more ones within that tensor.

The next step is to sift out too small probabilities (here 0.3 threshold was used). After that I applied
standard OpenCV findContours and boundingRect algorithms to find bounding boxes. This process is class independent - 
it is processed on each slice (according to last dimension) of the output tensor independently. 
Too small contours are rejected as false positives. If the objects of 
single class are too crowded, the bounding boxing algorithm treats them as
single objects - that's a flaw I am working on. 

One picture is worth a thousand words, so let's see this process on images.

1. Original image.

![alt text](repo_images/tennis_original.png)

2. Some of slices (masks) from generated tensor of probabilities. Note that thresholding has not yet been applied.

![alt text](repo_images/non_thresholded_tennis.png)

3. Nonzero masks after thresholding (the second mask is rejected in later part of pipeline, because its area is too small)

![alt text](repo_images/thresholded_tennis.png)

4. Final image with bounding boxes.

![alt text](repo_images/tennis_bboxes.png)

### Requirements
Used libraries (versions are provided only for informational purpose - the code should
run with no harm on reasonable younger/older versions)
- tensorflow 1.4.1
- numpy 1.14.1
- opencv 3.3.1

### How to run it?
1. 
    Create folder named 'data'. In order to get anything done, you first need to downloaded COCO data from here: http://cocodataset.org/#download . Download:
    - 2017 Train images (wget http://images.cocodataset.org/zips/train2017.zip)
    - 2017 Val images (wget http://images.cocodataset.org/zips/val2017.zip)
    - 2017 Train/Val annotations (wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
    - 2017 Test images (wget http://images.cocodataset.org/zips/test2017.zip) - unnecessary, but useful if you want to test it on COCO online evaluator (it yields bad results, however, due to weak clustering algorithm).
    - 2017 Testing image info (wget http://images.cocodataset.org/annotations/image_info_test2017.zip) - again, not needed if you don't want to test it on COCO evaluations script. If you download it, rename annotations inside to test_annotations. 
        
    Extract these files and put their content into previously created 'data' folder. Your project tree should look like this (most folders not shown for brewity):
    - data
        - annotations
        - train2017
        - val2017
        - test2017 (unnecessary)
        - test_annotations (unnecessary)
    
    End of step 1:) Remove downloaded zipped files if you don't need them anymore.
    
2. 
    Create folder named 'pretrained_imagenet' in your main project directory, download weights pretrained on ImageNet classification task from https://drive.google.com/open?id=1L6lwpORCHMbU6_eyIu9MWHlvVl12RQgT ,
    then put downloaded file into freshly created folder and rename the file to 'pretrained_imagenet'.
   

3. COCO Object Detection Dataset contains over 100k images with annotated
    objects belonging to 80 classes. The network outputs tensor of shape SxSxC. S states for an
    arbitrary size. You could provide independent datasets for different architectures.
    To create dataset, run prepare.py with parameters:
    - dst_w (width of generated labels)
    - dst_h (height of generated labels)
    - annotations_path (path to FILE with annotations - that's the one with JSON extension)
    - labels_path (path to folder, where labels should be placed - it will be created automatically if not present yet)
    
    YOU SHOULD GENERATE TRAIN AND VALIDATION SETS INDEPENDENTLY! Don't do anything with test data.
    Generally I used S = dst_w = dst_h. In case of generating training data, be prepared to very long wait - it tooks 5 hours on my laptop.
    
    Example of usage (the network is compatible with these params):
    - python prepare.py --dst_w=14 --dst_h=14 --annotations_path=data/annotations/instances_val2017.json --labels_path=data/val_labels_S14
    - python prepare.py --dst_w=14 --dst_h=14 --annotations_path=data/annotations/instances_train2017.json --labels_path=data/train_labels_S14
    
    Ok, enough preprocessing, let's train it!
    
4. If you want to train your network from scratch, run train.py with params:
   - model_name (name of model)  
   - epochs (number of epochs, recommended value = 100)
   - l_rate (learning rate, recommended value = 0.00001)
   - thresh (threshold, default value = 0.3)
   - batch_size (size of single batch, I used 15)
   - saver_checkpoint (save every n iterations, recommended value = 1500)
   - t_img_path (path to folder with train images)
   - v_img_path (path to folder with validation images)
   - t_label_path (path to folder with train labels)
   - v_label_path (path to folder with validation labels)
   
   It will train your network and, log progress and metrics into saved_summaries/model_name and save every n iterations into saved_models/model_name
   
   Example of usage: 
   python train.py --model_name=my_model --epochs=100 --l_rate=0.00001 --thresh=0.3 --batch_size=10 --saver_checkpoint=15 --t_img_path=data/train2017 --v_img_path=data/val2017 --t_label_path=data/train_labels_S14 --v_label_path=data/val_labels_S14

   If you want to continue training from previously train model, apart from aforementioned params,
   you should also provide checkpoint number:
   - load_checkpoint (number of checkpoint, multiply of saver_checkpoint)
   
5. If you want to predict bounding boxes of objects on images, first create folder 'sample_images' in your main project folder,
   then run test.py with params:
   - model_name (name of model)  
   - thresh (threshold, default value = 0.3)
   - model_checkpoint (number of checkpoint)
   
   If you want to run model pretrained on COCO, download it from here:
   https://drive.google.com/open?id=1_uECl9HKH-Bps6QTcPGa6rn21y97l7pP, extract and put into 'saved_models'.
   
   Example of usage:
   python test.py --model_name=model_s14 --model_checkpoint=62695 --thresh=0.3
   
6. If you want to test your model on COCO dataset, first you need to generate COCO file with COCO_test.py with params;
   - model_name (name of model)  
   - thresh (threshold, default value = 0.3)
   - model_checkpoint (number of checkpoint)
   - data_path (path to folder with test images)
   - annotations_path (path to test annotations)
   - show_images (wheter show or not images, rather for debugging purpose, values t/f)
   
   Example of usage:
   python COCO_test.py --model_name=model_s14 --model_checkpoint=62695 --data_path=data/test2017 --annotations_path=data/test_annotations/image_info_test-dev2017.json --thresh=0.3 --show_images=t

      
    
### Some results
![alt text](repo_images/0.jpg)
![alt text](repo_images/1.jpg)
![alt text](repo_images/6.jpg)
![alt text](repo_images/8.jpg)
![alt text](repo_images/49.jpg)
![alt text](repo_images/63.jpg)
![alt text](repo_images/82.jpg)
