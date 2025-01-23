


# Very-Deep-Convolutional-Networks-for-Large-Scale-Image-Recognition

This repository is a fork of [this work](https://github.com/Prabhu204/Very-Deep-Convolutional-Networks-for-Large-Scale-Image-Recognition/tree/master), a
implementation of [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) in Pytorch by [Prabhu Appalapuri](https://github.com/Prabhu204).

We made the models run on the CIFAR100 dataset, and modified them to support our different scheduling strategies.


### Usage

Very Deep Convolutional Networks for Large Scale Image Recognition
    
    [-h] [-t TRAIN] [-v VAL] [-b {64,128,256}] [-e {50,100,150}]
    [-d {11,13,16,19}] [-c11] [-es EARLY_STOPPING] [-i IMAGESIZE] [-lr LR]


 optional arguments:
 
      -h, --help            show this help message and exit
      -t TRAIN, --train TRAIN
                            required image dataset for training a model. It must
                            be in the data directory
      -v VAL, --val VAL     required image dataset for training a model. It must
                            be in the data directory
      -b {64,128,256}, --batchsize {64,128,256}
                            select number of samples to load from dataset
      -e {50,100,150}, --epochs {50,100,150}
      -d {11,13,16,19}, --depth {11,13,16,19}
                            depth of the deep learning model
      -c11, --conv1_1       setting it True will replace some of the 3x3 Conv
                            layers with 1x1 Conv layers in the 16 layer network
      -es EARLY_STOPPING, --early_stopping EARLY_STOPPING
                            early stopping is used to stop training of network, if
                            does not improve validation loss
      -i IMAGESIZE, --imagesize IMAGESIZE
                            it is used to resize the image pixels
      -lr LR, --lr LR       learning rate for an Adam optimizer
      -lf LOSSES_FILE, --losses_file LOSSES_FILE
      	                    Base of the name of the file in which the (pickled)
			    losses will be saved


<b>Example usage:</b>

For training a model having a layer depth of 11:
            
    python train.py -d 11 -e 50 -es 5 -b 50 
           
For training a model having a layer depth of 16 along with smaller Conv filter 1x1:
        
    python train.py -d 16 -c11 -e 50 -es 5 -b 50 

