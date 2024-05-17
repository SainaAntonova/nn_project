import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

class MyModel(nn.Module):
    def __init__(self, class_counts):
        super(MyModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, class_counts)

    def forward(self, x):
        return self.model(x)
    


# Загрузка модели
@st.cache_resource
def get_model():
    class_counts = 2  # Укажите количество классов
    model = MyModel(class_counts)
    model.load_state_dict(torch.load('cancer_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


def classify_image(image_path):
    model = get_model()
    image = Image.open(image_path).convert('RGB')
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[predicted_class].item()

st.title("Классификация рака кожи")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Изображение', use_column_width=True)

    predicted_class, probability = classify_image(uploaded_file)
    st.write("Предсказанная метка:", predicted_class)
    st.write("Вероятность:", probability * 100, "%")

