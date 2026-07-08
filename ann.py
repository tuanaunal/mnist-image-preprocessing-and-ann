import cv2  # opencv
import numpy as np  # sayisal islemler icin
import matplotlib.pyplot as plt  # gorsellestirme icin

from tensorflow.keras.datasets import mnist  # MNIST veri seti
from tensorflow.keras.models import Sequential  # ANN modeli icin
from tensorflow.keras.layers import Dense, Dropout  # ANN katmanlari icin
from tensorflow.keras.optimizers import Adam  # optimizer

print("ann")