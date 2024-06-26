import os
from PIL import Image


def save(image, directory, name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, name)
    image = Image.fromarray(image)
    image.save(filename)
