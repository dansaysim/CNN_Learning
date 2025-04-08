import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from config import config

#Load/download CIFAR10 dataset

full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

#Split into train and validation sets, 90% train, 10% validation(subject to change)

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [45000, 5000])

#Create DataLoaders

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, persistent_workers=True)

# Load test dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, persistent_workers=True)


