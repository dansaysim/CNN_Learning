import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from config import config
import sys
from torch.profiler import profile, record_function, ProfilerActivity

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_v1 import CNN_v1
from data_loader.data_loader import train_loader, val_loader, test_loader

#Set device and load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("Training on GPU")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU")

model = CNN_v1().to(device)

#Set loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Initialize history dictionary
history = {
    'train_loss': [],
    'train_accuracy': [],
    'val_loss': [],
    'val_accuracy': [],
    'test_loss': [],
    'test_accuracy': []
}

# Initialize training time tracking
total_start_time = time.time()

# Setup profiler output directory
profile_dir = os.path.expanduser("~/projects/CNN_Learning/profile_traces")
if not os.path.exists(profile_dir):
    os.makedirs(profile_dir)

# Define profiler configuration
profiler_schedule = torch.profiler.schedule(
    wait=1,        # Skip the first iteration
    warmup=1,      # Number of warmup iterations
    active=5,      # Number of iterations to profile
    repeat=1       # Repeat the profiling cycle
)

profiler_activities = [
    ProfilerActivity.CPU,
    ProfilerActivity.CUDA
]

# Create Holistic Trace handler function
def trace_handler(prof):
    trace_file = os.path.join(profile_dir, f"profile_trace_{time.strftime('%Y%m%d_%H%M%S')}.json")
    prof.export_chrome_trace(trace_file)
    print(f"Trace exported to: {trace_file}")
    
    # Print profiler summary to console
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Save detailed profiler table output
    table_file = os.path.join(profile_dir, f"profile_table_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(table_file, 'w') as f:
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    
    # Memory usage analysis
    memory_file = os.path.join(profile_dir, f"memory_stats_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(memory_file, 'w') as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=50))

# Setup profiler instance
profiler = torch.profiler.profile(
    activities=profiler_activities,
    schedule=profiler_schedule,
    on_trace_ready=trace_handler,
    record_shapes=True,
    profile_memory=False,
    with_stack=True
)

#Training loop

for epoch in range(config.epochs):
    model.train()
    train_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0
    
    # Start profiler only in the first epoch
    if epoch == 0:
        profiler.start()
    
    for i, (images, labels) in enumerate(train_loader, 0):
        with record_function("batch_processing"):
            inputs = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with record_function("optimizer_zero_grad"):
                optimizer.zero_grad()
                
            with record_function("forward"):
                outputs = model(inputs)
                
            with record_function("loss_calculation"):
                loss = criterion(outputs, labels)
                
            with record_function("backward"):
                loss.backward()
                
            with record_function("optimizer_step"):
                optimizer.step()

            train_loss += loss
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum()
        
        # Step the profiler only in the first epoch
        if epoch == 0:
            profiler.step()
            # Stop profiling after a certain number of iterations to avoid excessive data
            if i >= 50:
                profiler.stop()
                print(f"Profiling completed, traces saved to {profile_dir}")
                break
    
    train_loss = train_loss.item() / len(train_loader)
    train_accuracy = 100 * correct.item() / total

    #Validation loop
    model.eval()
    val_loss = torch.tensor(0.0, device=device)
    val_correct = torch.tensor(0, device=device)
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            inputs, labels = images.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum()

    val_loss = val_loss.item() / len(val_loader)
    val_accuracy = 100 * val_correct.item() / val_total

    #Test loop

    model.eval()
    test_loss = torch.tensor(0.0, device=device)
    test_correct = torch.tensor(0, device=device)
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            inputs, labels = images.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum()

    test_loss = test_loss.item() / len(test_loader)
    test_accuracy = 100 * test_correct.item() / test_total

    # Calculate total elapsed time
    
    total_time = int(time.time() - total_start_time)
    
    print(f"Epoch {epoch+1}/{config.epochs}: {total_time} seconds")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    #Save training history

    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)
    history['test_loss'].append(test_loss)
    history['test_accuracy'].append(test_accuracy)

# Save the final model

if not os.path.exists("model_outputs"):
    os.makedirs("model_outputs")
torch.save(model.state_dict(), "model_outputs/model_v1_final.pth")

# Calculate and print total training time
total_training_time = int(time.time() - total_start_time)
print(f"Total Training Time: {total_training_time} seconds")

# Plot final training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.legend()

plt.savefig("model_outputs/training_plots_final.png")
plt.close()
    