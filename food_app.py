import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# pip freeze > requirements.txt

# Baixar o modelo do Google Drive
model_path = 'model_after_fine_tuning_1.keras'

if not os.path.exists(model_path):
 gdown.download('https://drive.google.com/uc?id=1hxDY7WN23Doi6LvDjVoTH3GISaDXG28C', 'model_after_fine_tuning_1.keras', quiet=False)


model = load_model(model_path)
class_names = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheesecake',
 'cheese_plate',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']


# Função para processar a imagem
def preprocess_image(img):
    img = img.resize((480, 480))  # Redimensionar a imagem para o tamanho esperado pelo modelo
    img_array = np.expand_dims(img, axis=0)  # Adicionar uma dimensão extra
    return img_array


# Título do aplicativo
st.title("Classificador de Comida")

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Exibir a imagem carregada
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagem Carregada', use_column_width=True)

    # Processar a imagem
    img_array = preprocess_image(img)

    # Fazer a previsão
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Obter a classe prevista
    predicted_class_name = class_names[predicted_class[0]]

    # Exibir o resultado
    st.write(f"Classe Prevista: {predicted_class_name.title()}")  # Substitua pelo mapeamento de classes se necessário
