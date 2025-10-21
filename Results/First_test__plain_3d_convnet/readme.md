This is the first test done.

Once we knew that our training pipeline was working correctly, we proceeded to train the same network as before with more samples. The architecture of the network is the following:

![alt text](https://github.com/juannico007/DL-aneurysm-detection/blob/model-training-pytorch/Results/Network%20descriptions/first-tests.png)

The network is composed of 4 convolutional blocks of convolution, pooling, and batch normalization. Followed by a global average pooling and 2 dense layers for classification. 
Since the first problem to solve is a binary classification problem, the last layer is only one neuron.

At this first stage, we ran the model only over 182 images, having 126 images to train and 56 to validate.

During training, we moved the model to the GPU, using a batch size of 2, for 10 epochs; we tried to use a bigger batch size, but since the model is so big, only 2 images filled
the 8 GB of GPU ram, the training time was approximately 37 hours. 

At this stage, the weight and performance problems observed before became more evident and had to be addressed in the next iteration of the project in order to make more improvements.

The parameters used for the training were the following:

- BATCH_SIZE = 2
- EPOCHS = 50
- INITIAL_LEARNING_RATE = 0.0001
- EPOCHS_TO_LEARNING_RATE_REDUCTION_ON_PLATEAU = 4
- OPTIMIZER = ADAM

The model reduced the learning rate multiple times when hitting a plateau since the number of epochs for this was very low. After hitting a plateau, the learning rate was halved
The epochs in which this happened were 10, 16, 22, 28, 38, and 44.

The spiky behaviour of the loss curve can be explained by 2 possible factors: First, the initial learning rate can be very high. Second, the batch size is too low.

Also, having a batch size of only two images makes our batch normalization layers almost useless, so we needed to find a way to train with a bigger batch size.

Looking at the model, we found that the final convolutional layer, where the number of filters went up from 128 to 256 is the main bottleneck, where most of the parameters of the network are acumulated.
This is why we decided to modify the number of filters of each convolutional layer, halving them, starting with 32 filters after the first convolution and going up to 128 for our next iteration.

The effect of this will be making the model lighter, allowing to train on a bigger batch size and also to make the model lighter, allowing us to train with more samples in future iterations.
