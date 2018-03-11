### Description
This is a TensorFlow implementation of simple yet powerful object detector, 
based on autoencoder architecture. I actually test 3 different architectures:
- plain convolutional network, generating tensor of shape SxSxC (C - number of object classes)
- autoencoder
- variational autoencoder.

The proposed solution is almost end to end - due to probabilistic nature of generated
tensors, they requires thresholding and blob finding.

### Requirements
Used libraries (versions are provided only for informational purpose - the code should
run with no harm on reasonable younger/older versions)
- tensorflow 1.4.1
- numpy 1.14.1
- opencv 3.3.0

### Running steps
1. 
    Create folder named 'data'. In order to get anything done, you first need to downloaded COCO data from here: http://cocodataset.org/#download . Download:
    - 2017 Train images
    - 2017 Val images
    - 2017 Test images
    - 2017 Train/Val annotations
    
    Extract these 4 files and put their content into previously created 'data' folder. Your project tree should look like this (most folders not shown for brewity):
    - data
        - annotations
        - test2017
        - train2017
        - val2017
    
    End of step 1:) Remove downloaded zipped files if you don't need them anymore.

2. COCO Object Detection Dataset contains over 100k images with annotated
    objects belonging to 80 classes. The network outputs tensor of shape SxSxC. S states for an
    arbitrary size. You could provide independent datasets for different architectures.
    To create dataset, run prepare.py with parameters:
    - dst_w (width of generated labels)
    - dst_h (height of generated labels)
    - annotations_path (path to FILE with annotations - that's the one with JSON extension)
    - labels_path (path to folder, where labels should be placed - it will be created automatically if not present yet)
    
    YOU SHOULD GENERATE TRAIN AND VALIDATION SETS INDEPENDENTLY! Don't do anything with test data, because there are no annotations for this data.
    Generally I used S = dst_w = dst_h. In case of generating training data, be prepared to very long wait - it tooks 5 hours on my laptop.
    
    Example of usage:
    - python prepare.py --dst_w=14 --dst_h=14 --annotations_path=data/annotations/instances_val2017.json --labels_path=data/labels_S14
     

