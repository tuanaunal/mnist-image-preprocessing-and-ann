"""
MNIST veri seti:
    rakamlama: 0-9 toplamda 10 sinif var
    28*28 piksel boyutunda resimler
    grayscale (siyah beyaz) resimler
    60000 egitim, 10000 test verisi
    amacimiz: ann ile bu resimleri tanimlamak ya da siniflandirmak

Image processing:
    histogram esitleme: kontrast iyilestirme
    gaussian blur: gurultu azaltma
    canny edge detection:kenar tespiti

    ANN (Artificial Neural Network) ile MNIST veri setini siniflandirma

    libraries:
        tensorflow: keras ile ANN modeli olusturma ve egitim
        matplotlib: gorsellestirme
        cv2: opencv image processing
"""


# import libraries
import cv2  # opencv
import numpy as np  # sayisal islemler icin
import matplotlib.pyplot as plt  # gorsellestirme icin

from tensorflow.keras.datasets import mnist  # MNIST veri seti
from tensorflow.keras.models import Sequential  # ANN modeli icin
from tensorflow.keras.layers import Dense, Dropout  # ANN katmanlari icin
from tensorflow.keras.optimizers import Adam  # optimizer

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# image preprocessing
img = x_train[0] #ilk resmi al

stages = {"orijinal": img}

# histogram eşitleme
eq = cv2.equalizeHist(img) # histogram eşitleme
stages["Histogram Eşitleme"] = eq

# gaussian blur: gurultuyu azaltma
blur = cv2.GaussianBlur(eq, (5, 5), 0)
stages["gaussian blur"] = blur

# canny ile kenar tespiti
edges = cv2.Canny(blur, 50, 150) # kenar tespiti
stages["canny kenarlari"] = edges

# gorsellestirme
fig, axes = plt.subplots(2, 2, figsize=(6,6))
axes = axes.flat
for ax, (title, im) in zip(axes, stages.items()):
    ax.imshow(im, cmap ="gray")
    ax.set_title(title)
    ax.axis("off")

plt.suptitle("MNIST Image Preprocessing Stages")
plt.tight_layout()
plt.show()

# preprocess fonksiyonu
def preprocess_image(img):
    """
    - histogram esitleme
    - gaussian blur
    - canny ile kenar tespiti
- flattering: 28*28 boyutundan 784 boyutuna cevirme
- normalizasyon: 0-255 arasindan 0-1 arasina cevirme
    """
    img_eq = cv2.equalizeHist(img) # histogram esitleme
    img_blur = cv2.GaussianBlur(img_eq, (5,5), 0) #gaussian blur
    img_edges = cv2.Canny(img_blur, 50, 150) #canny kenar tespiti
    feature = img_edges.flatten() / 255.0 # flattering: 28*28 boyutundan 784 boyutuna cevirme
    return feature

num_train =10000
num_test = 2000

x_train = np.array([preprocess_image(img) for img in x_train[ :num_train]])
y_train_sub = y_train[ :num_train]

x_test = np.array([preprocess_image(img) for img in x_test[ :num_test]])
y_test_sub = y_test[ :num_test]

# ann model creation
model = Sequential([
Dense(128, activation = "relu", input_shape = (784,)), # ilk katman, 128 nöron 28*28 = 784 boyutunda
Dropout(0.5), # dropout katmani, overfitting'i azaltmak icin, %50 dropout
Dense(64, activation = "relu"), # ikinci katman, 64 nöron
Dense(10, activation = "softmax") #cikis katmani, 10 nöron (0-9 rakamlari icin)
])

# compile model
model.compile(
    optimizer = Adam(learning_rate = 0.001), # optimizer
    loss = "sparse_categorical_crossentropy", # kayip fonksiyonu
    metrics = ["accuracy"] # metrikler
)

print(model.summary())

# ann model training
history = model.fit(
    x_train, y_train_sub,
    validation_data = (x_test, y_test_sub),
    epochs = 10, #epoch sayisi
    batch_size = 32, # batch boyutu
    verbose = 2
)

# evaluate model performance
