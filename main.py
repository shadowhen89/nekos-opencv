import os
import minerals
import random
import cv2
import matplotlib.pyplot as plt

IMG_EXT = ['jpg', 'png']
IMG_FOLDER = 'images'


def get_images(dir_path):
    files = os.listdir(dir_path)
    images = []
    for f in files:
        if f.split('.')[-1] in IMG_EXT:
            images.append(f)
    return images


def main():
    if not os.path.exists(IMG_FOLDER):
        print('Folder \"%s\" is not found. Creating a folder...' % IMG_FOLDER)
        os.makedirs(IMG_FOLDER)
    else:
        print('Folder \"%s\" is found. Accessing the folder now...' % IMG_FOLDER)
        images = get_images(IMG_FOLDER)

        if len(images) > 0:
            print("Images found. Randomizing the choice...")

            # Randomizes the choice of the picture
            image_path = 'images' + os.sep + images[random.randint(0, len(images) - 1)]
            print('Randomization done. ' + image_path + ' is the image for processing')

            image = cv2.imread(image_path)
            result = minerals.process_gold_silver(image)

            # Shows the image on the pyplot
            plt.imshow(result)
            plt.show()
        else:
            print("No images. Canceling the process...")


if __name__ == '__main__':
    main()
