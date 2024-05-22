import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    

def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (224, 224))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    image_array = img_to_array(image) / 255.0
    return image_array

datagen = ImageDataGenerator(rescale=1.0/255)

#def preprocess_image(image, sigmaX=10):
 #   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #  image = crop_image_from_gray(image)
   # image = cv2.resize(image, (224, 224))  # Cambia el tamaño según las necesidades de tu modelo
   # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
   # return image

def load_image(image_file):
    img = Image.open(image_file)
    return img


#model = load_model("path_to_your_model.h5")  # Cambia a la ruta de tu modelo
#url= "https://drive.google.com/file/d/1ELGWX058ElB9eAe-MbfMuIUgUAEVdVP4/view?usp=sharing"
#output= "Modelo_DesNet121"
#gdown.download(url, output, quiet=False)
#model = load_model(output)