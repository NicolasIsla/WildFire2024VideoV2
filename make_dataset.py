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

# Configuración de argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Procesar videos con YOLO y extracción de características.')
parser.add_argument('--yolo_path', type=str, required=True, help='Ruta al modelo YOLO preentrenado.')
parser.add_argument('--data_path', type=str, required=True, help='Ruta a los datos de entrada.')
parser.add_argument('--output_path', type=str, required=True, help='Ruta de salida para guardar los resultados.')
parser.add_argument('--buffer_size', type=int, default=5, help='Tamaño del buffer para el dataset de video.')
parser.add_argument('--iou', type=float, default=0.01, help='Umbral de IoU para considerar una detección válida.')
args = parser.parse_args()

# Definición de rutas
path = args.data_path
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

# Cargar el modelo YOLO
model_yolo = YOLO(args.yolo_path)

# Configuración de transformaciones
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

to_pil = ToPILImage()

# Funciones de utilidad

def calculate_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0


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

# Clase Dataset personalizada
class VideoDatasetWithBuffer(Dataset):
    def __init__(self, frames, labels, buffer_size=5, yolo=None, iou=0.01, resize=(640, 640)):
        self.frames = frames
        self.labels = labels
        self.buffer_size = buffer_size
        self.yolo = yolo
        self.iou = iou
        self.resize = resize

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        start_index = max(0, index - self.buffer_size + 1)
        buffer = self.frames[start_index:index + 1]
        labels = self.labels[start_index:index + 1]

        if len(buffer) < self.buffer_size:
            padding = torch.zeros(self.buffer_size - len(buffer), *buffer[0].shape)
            buffer = torch.cat((padding, buffer), dim=0)
            padding = torch.ones(self.buffer_size - len(labels), *labels[0].shape) * -1
            labels = torch.cat((padding, labels), dim=0)

        yolo_output = self.yolo.predict(buffer[-1].unsqueeze(0), conf=0.001) if self.yolo else None
        boxes, labels_chopped = [], []

        if yolo_output:
            for result in yolo_output:
                for box in result.boxes.xyxyn:
                    x1, y1, x2, y2 = box[:4]
                    if (x2 - x1) * (y2 - y1) * self.resize[0] * self.resize[1] < 20:
                        continue

                    detections = [calculate_iou([x1, y1, x2, y2], label) > self.iou for label in labels]
                    x1, y1, x2, y2 = x1 * self.resize[0], y1 * self.resize[1], x2 * self.resize[0], y2 * self.resize[1]
                    y1, y2, x1, x2 = make_square_chop(y1, y2, x1, x2)

                    chop = buffer[:, :, int(y1):int(y2), int(x1):int(x2)]
                    chop = torch.nn.functional.interpolate(chop, self.resize)
                    labels_chopped.append(torch.tensor(detections))
                    boxes.append(chop)

        if not boxes:
            return None, None

        return torch.stack(boxes), torch.stack(labels_chopped)

# Clase para la extracción de características
class ImageFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, output_features=256):
        super(ImageFeatureExtractor, self).__init__()

        if model_name == 'resnet50':
            self.feature_extractor = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            feature_dim = 2048

        elif model_name == 'efficientnet_b0':
            self.feature_extractor = models.efficientnet_b4(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            feature_dim = 1792

        else:
            raise ValueError(f"Modelo '{model_name}' no soportado.")

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(feature_dim, output_features)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)

        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        out = features.view(batch_size, sequence_length, -1)

        return out

# Cargar frames y bounding boxes

def cargar_frames(carpeta):
    frames, bounding_boxes = [], []
    for frame in sorted(os.listdir(carpeta)):
        if frame.endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(carpeta, frame)).convert('RGB')
            frames.append(transform(img))

            label_path = os.path.join(carpeta, 'labels', frame.replace('.jpg', '.txt'))
            if os.path.exists(label_path):
                with open(label_path) as f:
                    for line in f:
                        class_id, x_centro, y_centro, ancho, alto = map(float, line.strip().split())
                        x_min, y_min = x_centro - ancho / 2, y_centro - alto / 2
                        x_max, y_max = x_centro + ancho / 2, y_centro + alto / 2
                        bounding_boxes.append([x_min, y_min, x_max, y_max])
            else:
                bounding_boxes.append([-1, -1, -1, -1])

    return torch.stack(frames), torch.tensor(bounding_boxes, dtype=torch.float32)

# Procesar carpetas y guardar chops
resnet50 = ImageFeatureExtractor(model_name='resnet50')
efficientnet_b0 = ImageFeatureExtractor(model_name='efficientnet_b0')
models = {'resnet50': resnet50, 'efficientnet_b0': efficientnet_b0}

# Procesar y guardar los resultados
from tqdm import tqdm

carpetas = [carpeta for carpeta in os.listdir(path) if os.path.isdir(os.path.join(path, carpeta))]
contador =0
for carpeta in tqdm(carpetas, desc="Procesando carpetas"):
    carpeta_path = os.path.join(path, carpeta)
    frames, bounding_boxes = cargar_frames(carpeta_path)
    dataset = VideoDatasetWithBuffer(
        frames,
        bounding_boxes,
        buffer_size=args.buffer_size,
        yolo=model_yolo,
        iou=args.iou
    )

    # Filtrar dataset para evitar datos nulos
    data = [(boxes, labels) for boxes, labels in dataset if boxes is not None and labels is not None]
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    for chop_boxes, labels in tqdm(data_loader, desc=f"Procesando {carpeta}", leave=False):
        contador += 1
        torch.save(labels[0], os.path.join(output_path, f"labels_{contador}.pt"))
        for model_name, model in models.items():
            features = model(chop_boxes[0])
            torch.save(features, os.path.join(output_path, f"{model_name}_{contador}.pt"))

print("Proceso completado.")
