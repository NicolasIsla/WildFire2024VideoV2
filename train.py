import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchcrf import CRF
import torch.optim as optim

# Define device configuration
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load the dataset
final_dataset_path = "features.pt"
chops = torch.load(final_dataset_path)
final_dataset_path = "labels.pt"

print(chops.shape)
labels = torch.load(final_dataset_path)
print(labels.shape)
