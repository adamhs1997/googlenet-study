"""
NOTE: GNet works on the following preprocessing fcn:
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
Obtained from https://stackoverflow.com/questions/44341258/preprocessing-function-of-inception-v3-in-keras
"""


import foolbox
import keras
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from skimage.io import imsave

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = InceptionV3(weights='imagenet')
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 1), preprocessing=(0.5, 0.5))

# get source image and label
image, label = foolbox.utils.imagenet_example()

# Bound at 0-1 for GNet
image /= 255

# see predictions
print("corrrect", label)
print("this", np.argmax(fmodel.predictions(image)))

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image, label)
# if the attack fails, adversarial will be None and a warning will be printed
print(adversarial)
imsave("original.png", image, plugin='pil', format_str='png')
imsave("adversarial.png", adversarial, plugin='pil', format_str='png')
