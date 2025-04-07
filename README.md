### CNN_v1
Convolution Neural Net Practice

This is a repo where I will be implementing a Keras-free CNN. 

The plan is to start very basic and then incrementally add new feauters to speed up training/inference/increase accuracy.

In between each feature addition the model will be re-trained and benchmarked(starting very small for fast training times).

The goal is to have hands on experience working and improving upon a CNN.

## Training the Model

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Start training:
```
python -m training.train
```

The training script will:
- Load the CIFAR-10 dataset
- Train the CNN model for 20 epochs by default
- Display training, validation, and test metrics for each epoch
- Save model checkpoints after each epoch
- Generate training plots showing loss and accuracy trends

You can modify training parameters in `config.py`.