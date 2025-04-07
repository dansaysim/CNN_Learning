##### CNN

#### Models

### CNN_v1 

2 Convolution Layers
    - 3,16,3,1
    - 16,64,3,1
2 ReLu Activations
2 MaxPool Layers
    - 2,2
1 FC Layer
    - 4096,10

Conv->ReLu->MaxPool (2x) -> FC

## Results:
(On Laptop, forgot to add training time calc, do again with time and compare)

# Laptop: CNN v1 - Test 1

Epochs: 50
Batch_Size: 512
Learning_Rate: 1e-3
Classes = 10
Training/Validation Data Split: 90/10

Final Epoch:

Epoch 50/50, Train Loss: 0.6240, Train Accuracy: 78.55%, Val Loss: 0.9156, Val Accuracy: 69.54%
Test Loss: 0.9221, Test Accuracy: 69.24%

Best Accuracy:

Epoch 43/50, Train Loss: 0.6641, Train Accuracy: 77.39%, Val Loss: 0.8961, Val Accuracy: 69.54%
Test Loss: 0.8968, Test Accuracy: 69.53%

Thoughts: Model began to overfit around epoch ~25, worsening as time went on

# Laptop: CNN v1 - Test 2

Epochs: 50
Batch_Size: 512
Learning_Rate: 1e-3
Classes = 10
Training/Validation Data Split: 80/20

Final Epoch:

Epoch 50/50, Train Loss: 0.6278, Train Accuracy: 78.65%, Val Loss: 0.9871, Val Accuracy: 67.60%
Test Loss: 0.9587, Test Accuracy: 68.26%

Best Accuracy:

Epoch 46/50, Train Loss: 0.6699, Train Accuracy: 77.51%, Val Loss: 0.9408, Val Accuracy: 68.55%
Test Loss: 0.9160, Test Accuracy: 69.32%

Thoughts: Model began to overfit around epoch ~20, very jagged validation loss




#### Training the Model

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