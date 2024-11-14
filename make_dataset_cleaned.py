# Importaciones estándar
import os
import random

# Importaciones de terceros
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import ToPILImage
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import argparse
import pandas as pd

data = pd.read_csv('data_temp_cleaned.csv')

dataset_path = "/data/nisla/data/train/"

def make_square_chop(y1, y2, x1, x2, img_height=640, img_width=640, growth_factor=1.1):
    center_y = (y1 + y2) / 2
    center_x = (x1 + x2) / 2
    half_size = max((y2 - y1), (x2 - x1)) / 2 * growth_factor

    half_size = min(half_size, center_y, img_height - center_y, center_x, img_width - center_x)
    new_y1 = max(0, int(center_y - half_size))
    new_y2 = min(img_height, int(center_y + half_size))
    new_x1 = max(0, int(center_x - half_size))
    new_x2 = min(img_width, int(center_x + half_size))

    return new_y1, new_y2, new_x1, new_x2


import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import cv2
import numpy as np
from tqdm import tqdm

# Clase para la extracción de características
class ImageFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, output_features=256):
        super(ImageFeatureExtractor, self).__init__()

        if model_name == 'resnet50':
            self.feature_extractor = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1]).to(device)
            feature_dim = 2048

        elif model_name == 'efficientnet_b0':
            self.feature_extractor = models.efficientnet_b4(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1]).to(device)
            feature_dim = 1792

        else:
            raise ValueError(f"Modelo '{model_name}' no soportado.")

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(feature_dim, output_features)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)

        features = self.feature_extractor(x.to(device))
        features = torch.flatten(features, 1)
        out = features.view(batch_size, sequence_length, -1)
        return out.to('cpu')

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instancia del extractor de características
extractor = ImageFeatureExtractor(model_name='resnet50', pretrained=True, output_features=256).to(device)

# Listas para almacenar características y etiquetas
features_list = []
labels_list = []
buffers_list = []

#  Procesar videos y extraer características
n_videos = 0
for video in tqdm(data["video"].unique(), desc="Procesando videos", unit="video"):
    print(f"Procesando video: {video}")
    frames = os.listdir(dataset_path + "/" + video)
    frames.sort()
    dff = data[data["video"] == video]
    n_videos += 1

    
    with tqdm(total=len(dff), desc=f"Frames en {video}", unit="frame") as pbar:
        for _, row in dff.iterrows():
            frame = row["frame"]
            box = row["box"]  # Formato [x1, y1, x2, y2]
            box = list(map(float, box[1:-1].split(", ")))

            labels = [row["t_4"], row["t_3"], row["t_2"], row["t_1"], row["t_0"]]
            
            # Encontrar la posición del frame actual en la lista de frames
            try:
                current_index = frames.index(frame)
            except ValueError:
                print(f"Frame {frame} no encontrado en la lista de frames para el video {video}.")
                pbar.update(1)
                continue
            
            # Crear un buffer con los 4 frames anteriores más el actual
            buffer = []
            for i in range(4, -1, -1):  # Desde el frame actual hasta 4 frames atrás
                frame_index = current_index - i
                if frame_index < 0 or frame_index >= len(frames):  # Índice inválido
                    buffer.append(np.zeros((640, 640, 3), dtype=np.uint8))  # Imagen vacía
                else:
                    img = cv2.imread(dataset_path + "/" + video + "/" + frames[frame_index])
                    if img is None:
                        buffer.append(np.zeros((640, 640, 3), dtype=np.uint8))
                    else:
                        img_height, img_width, _ = img.shape
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * img_width)
                        x2 = int(x2 * img_width)
                        y1 = int(y1 * img_height)
                        y2 = int(y2 * img_height)
                        cropped_img = img[y1:y2, x1:x2]
                        buffer.append(cropped_img)

            # Transformar las imágenes del buffer
            buffer = [transform(img) for img in buffer]
            buffer_tensor = torch.stack(buffer).to(device)  # (seq_len, C, H, W)
            buffer = torch.stack(buffer).unsqueeze(0).to(device)  # Batch x Sequence x C x H x W
            buffers_list.append(buffer_tensor.cpu())  # Guardar en la lista, pasando a CPU si es necesario

            # Extraer características
            with torch.no_grad():
                features = extractor(buffer)

            # Guardar características y etiquetas
            features_list.append(features.squeeze(0))  # Sin batch dimension
            labels_list.append(torch.tensor(labels))

            pbar.update(1)  # Actualizar la barra de progreso para f
    if n_videos == 400:
        break

# Convertir listas a tensores
features_tensor = torch.stack(features_list)  # Tensor de características
labels_tensor = torch.stack(labels_list)     # Tensor de etiquetas

# Guardar en archivo binario
output_path_features = "features.pt"
output_path_labels = "labels.pt"
torch.save(features_tensor, output_path_features)
torch.save(labels_tensor, output_path_labels)
torch.save(buffers_list, "buffers.pt")

print(f"Características guardadas en: {output_path_features}")
print(f"Etiquetas guardadas en: {output_path_labels}")
