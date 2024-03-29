# CapsNet

This repository provides Capsule Layer Implementations according to the original paper. Working in TensorFlow 2.x.

Configured to be installable as a pip package

# Files

* capsnet.py - Capsule Layer implementations usable to create your own models
* main.py - Code to train and evaluate the MNIST capsule network in the original paper
* best_weights.hdf5 - Pre-trained weights for the MNIST capsule network in main.py

## MNIST Capsule Network

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input (InputLayer)              [(None, 28, 28, 1)]  0                                            
__________________________________________________________________________________________________
conv (Conv2D)                   (None, 20, 20, 256)  20992       input[0][0]                      
__________________________________________________________________________________________________
conv_caps (CapsConv2D)          (None, 6, 6, 32, 8)  5308672     conv[0][0]                       
__________________________________________________________________________________________________
dense_caps (CapsDense)          (None, 10, 16)       1474560     conv_caps[0][0]                  
__________________________________________________________________________________________________
masking (Lambda)                (None, 10, 16)       0           dense_caps[0][0]                 
__________________________________________________________________________________________________
flatten (Flatten)               (None, 160)          0           masking[0][0]                    
__________________________________________________________________________________________________
decoder_l1 (Dense)              (None, 512)          82432       flatten[0][0]                    
__________________________________________________________________________________________________
decoder_l2 (Dense)              (None, 1024)         525312      decoder_l1[0][0]                 
__________________________________________________________________________________________________
decoder_l3 (Dense)              (None, 784)          803600      decoder_l2[0][0]                 
__________________________________________________________________________________________________
margin (Lambda)                 (None, 10)           0           dense_caps[0][0]                 
__________________________________________________________________________________________________
reconstruction (Reshape)        (None, 28, 28, 1)    0           decoder_l3[0][0]                 
==================================================================================================
Total params: 8,215,568
Trainable params: 8,215,568
Non-trainable params: 0
__________________________________________________________________________________________________
```

# Test Results

```
__________________________________________________________________________________________________
10000/10000 [==============================] - 31s 3ms/sample - loss: 0.0010 - margin_loss: 8.3830e-04 - reconstruction_loss: 0.0391 - margin_accuracy: 0.9919
__________________________________________________________________________________________________
```
