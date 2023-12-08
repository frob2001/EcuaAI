import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

# Definición de tu modelo StableDiffusion
class StableDiffusion(nn.Module):
    def __init__(self, image_size, num_channels, num_steps):
        super(StableDiffusion, self).__init__()

        self.num_steps = num_steps
        self.diffusion_model = nn.ModuleList([
            DiffusionStep(image_size, num_channels) for _ in range(num_steps)
        ])

    def forward(self, x):
        for diffusion_step in self.diffusion_model:
            x = diffusion_step(x)
        return x

class DiffusionStep(nn.Module):
    def __init__(self, image_size, num_channels):
        super(DiffusionStep, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        noise = torch.randn_like(x)
        x = x + noise
        x = self.net(x)
        return x

# Configuración de parámetros del modelo
image_size = (64, 64)
num_channels = 3
num_steps = 10

# Crea una instancia del modelo de difusión estable
model = StableDiffusion(image_size, num_channels, num_steps)

# Función para descargar y transformar una imagen
def load_and_transform_image(url, image_size):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    return transform(image)

# Lee el archivo CSV y procesa las imágenes
def process_images_from_csv(csv_file, model):
    df = pd.read_csv(csv_file)
    urls = df['urls'].apply(lambda x: eval(x)['regular'])
    transformed_images = [load_and_transform_image(url, image_size) for url in urls]

    for img_tensor in transformed_images:
        output_image = model(img_tensor.unsqueeze(0)) # Añade una dimensión de lote
        show_image(output_image)

# Función para visualizar una imagen generada
def show_image(tensor_image):
    generated_image = tensor_image.squeeze().detach().permute(1, 2, 0).numpy()
    plt.imshow(generated_image)
    plt.show()

# Llamada a la función de procesamiento de imágenes
process_images_from_csv('datos_limpieza.csv', model)
