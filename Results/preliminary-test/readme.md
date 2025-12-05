This is the first preliminary test done.

The purpose of this test was to assess the behavior of the training pipeline and ensure that all processes were running correctly without throwing any errors. Thus, we defined the initial architecture of our network, a 3D convolutional network with the following layers:

![alt text](https://github.com/juannico007/DL-aneurysm-detection/blob/model-training-pytorch/Results/Network%20descriptions/first-tests.png)

For these first tests, we also decided to use a fixed input size; we set a value of 256x256x256 for each volume

At this first stage, we ran the model only over 20 images, having 14 images to train and 6 to validate, since we didn't care much about the performance of our network.

During training, we moved the model to the GPU, using a batch size of 2, for 10 epochs; the training time was approximately 50 minutes. At this stage, we could start observing weight and performance problems that will be addressed in the next iterations of the project.

The parameters used for the training were the following:

- BATCH_SIZE = 2
- EPOCHS = 10
- INITIAL_LEARNING_RATE = 0.0001
- EPCOHS_TO_LEARNING_RATE_REDUCTION_ON_PLATEAU = 4
- OPTIMIZER = ADAM

As these results don't reflect at all the performance of the network, we needed to address this with a bigger training and validation set.
