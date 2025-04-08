# CNN

## Models

### CNN_v1 

2 Convolution Layers
- 3,16,3,1
- 16,64,3,1

2 ReLu Activations

2 MaxPool Layers
- 2,2

1 FC Layer
- 4096,10

{Conv->ReLu->MaxPool}(2x)->Flatten->FC

Afrt

#### Results:

###### Laptop: CNN v1 - Test 1

Epochs: 50 |
Batch_Size: 512 |
Learning_Rate: 1e-3 |
Classes = 10 |
Training/Validation Data Split: 90/10 

Final Epoch:

Epoch 50/50, Train Loss: 0.6240, Train Accuracy: 78.55%, Val Loss: 0.9156, Val Accuracy: 69.54%
Test Loss: 0.9221, Test Accuracy: 69.24%

Best Accuracy:

Epoch 43/50, Train Loss: 0.6641, Train Accuracy: 77.39%, Val Loss: 0.8961, Val Accuracy: 69.54%
Test Loss: 0.8968, Test Accuracy: 69.53%

Thoughts: Model began to overfit around epoch ~25, worsening as time went on

###### Laptop: CNN v1 - Test 2

Epochs: 50 |
Batch_Size: 512 |
Learning_Rate: 1e-3 |
Classes = 10 |
Training/Validation Data Split: 80/20

Final Epoch:

Epoch 50/50, Train Loss: 0.6278, Train Accuracy: 78.65%, Val Loss: 0.9871, Val Accuracy: 67.60%
Test Loss: 0.9587, Test Accuracy: 68.26%

Best Accuracy:

Epoch 46/50, Train Loss: 0.6699, Train Accuracy: 77.51%, Val Loss: 0.9408, Val Accuracy: 68.55%
Test Loss: 0.9160, Test Accuracy: 69.32%

Thoughts: Model began to overfit around epoch ~20, very jagged validation loss

###### Laptop: CNN v1 - Test 3

Epochs: 50 |
Batch_Size: 512 |
Learning_Rate: 1e-3 |
Classes = 10 |
Training/Validation Data Split: 90/10

Final Epoch:

Epoch 50/50: 396 seconds
Train Loss: 0.6561, Train Accuracy: 77.67%, Val Loss: 0.9060, Val Accuracy: 69.30%
Test Loss: 0.9226, Test Accuracy: 69.02%

Best Accuracy:

Epoch 48/50: 380 seconds
Train Loss: 0.6604, Train Accuracy: 77.45%, Val Loss: 0.8925, Val Accuracy: 68.90%
Test Loss: 0.9101, Test Accuracy: 69.44%

Thoughts: Model began to overfit around epoch ~20, slightly under 10 seconds per epoch. This is enough random tests that I will now swap to CNN v2 and begin to add dropout layers gradually.

### CNN_v2 - Added Dropout 

2 Convolution Layers
- 3,16,3,1
- 16,64,3,1

2 ReLu Activations

2 MaxPool Layers
- 2,2

3 Dropout Layers
- 25% after Conv Blocks
- 50% after Flatten

1 FC Layer
- 4096,10

{Conv->ReLu->MaxPool->Dropout}(2x)->Flatten->Dropout->FC

#### Results:

###### Laptop: CNN v2 - Test 1

Epochs: 50 |
Batch_Size: 512 |
Learning_Rate: 1e-3 |
Classes = 10 |
Training/Validation Data Split: 90/10

Final Epoch:

Epoch 50/50: 400 seconds
Train Loss: 0.9513, Train Accuracy: 66.82%, Val Loss: 0.9156, Val Accuracy: 69.10%
Test Loss: 0.8940, Test Accuracy: 70.01%

Best Accuracy:

Epoch 49/50: 393 seconds
Train Loss: 0.9502, Train Accuracy: 66.80%, Val Loss: 0.9096, Val Accuracy: 69.54%
Test Loss: 0.8879, Test Accuracy: 70.35%

Thoughts: No/very little overfitting after adding 3 dropout layers, going to increase epoch count to 100 and see if it continues to improve.

###### Laptop: CNN v2 - Test 2

Epochs: 100 |
Batch_Size: 512 |
Learning_Rate: 1e-3 |
Classes = 10 |
Training/Validation Data Split: 90/10

Final Epoch:

Epoch 100/100: 809 seconds
Train Loss: 0.8972, Train Accuracy: 68.76%, Val Loss: 0.7830, Val Accuracy: 74.14%
Test Loss: 0.8293, Test Accuracy: 72.36%

Best Accuracy:

Epoch 100/100: 809 seconds
Train Loss: 0.8972, Train Accuracy: 68.76%, Val Loss: 0.7830, Val Accuracy: 74.14%
Test Loss: 0.8293, Test Accuracy: 72.36%

Thoughts: Continued to improve past 50 epochs, best result was at 100 epochs so will still be adding more epochs but first need to optimize data loader to speed up training(hopefully)

###### Laptop: CNN v2 - Test 3

Epochs: 100 |
Batch_Size: 512 |
Learning_Rate: 1e-3 |
Classes = 10 |
Num_Workers = 14 |
Training/Validation Data Split: 90/10

Final Epoch:

poch 100/100: 198 seconds
Train Loss: 0.8841, Train Accuracy: 69.28%, Val Loss: 0.8391, Val Accuracy: 71.92%
Test Loss: 0.8495, Test Accuracy: 71.71%

Best Accuracy:

Epoch 93/100: 184 seconds
Train Loss: 0.8956, Train Accuracy: 68.67%, Val Loss: 0.8187, Val Accuracy: 72.30%
Test Loss: 0.8325, Test Accuracy: 72.00%

Thoughts: Ran  num_workers benchmark on laptop and got 14 as the fastest so used that. Dropped epoch time significantly. Next test will be on desktop but with num_workers optimized for that system.

###### Desktop: CNN v2 - Test 4

Epochs: 100 |
Batch_Size: 512 |
Learning_Rate: 1e-3 |
Classes = 10 |
Num_Workers = 8 |
Training/Validation Data Split: 90/10

Final Epoch:

Epoch 100/100: 63 seconds
Train Loss: 0.8747, Train Accuracy: 69.57%, Val Loss: 0.8033, Val Accuracy: 72.48%
Test Loss: 0.8259, Test Accuracy: 72.54%

Best Accuracy:

Epoch 100/100: 63 seconds
Train Loss: 0.8747, Train Accuracy: 69.57%, Val Loss: 0.8033, Val Accuracy: 72.48%
Test Loss: 0.8259, Test Accuracy: 72.54%

Thoughts: Everything kept the same except dropping num_workers to 8 but still led to a >3x speedup in training. Now going to significantly up Epoch and see how far I can push it before it begins overfitting.

###### Desktop: CNN v2 - Test 5

Epochs: 250 |
Batch_Size: 512 |
Learning_Rate: 1e-3 |
Classes = 10 |
Num_Workers = 8 |
Training/Validation Data Split: 90/10

Final Epoch:

Epoch 250/250: 164 seconds
Train Loss: 0.8101, Train Accuracy: 71.80%, Val Loss: 0.7741, Val Accuracy: 73.96%
Test Loss: 0.7961, Test Accuracy: 73.41%

Best Accuracy:

Epoch 235/250: 153 seconds
Train Loss: 0.8094, Train Accuracy: 71.80%, Val Loss: 0.7450, Val Accuracy: 75.22%
Test Loss: 0.7706, Test Accuracy: 74.86%

Thoughts: Loss starts to plateau slightly past 200 so I'll keep it maxed at 250 epochs. Now going to implement more features and swap to v3.

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