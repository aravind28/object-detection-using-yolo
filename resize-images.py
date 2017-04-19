import glob
import cv2

def process(filename):
    # split the path of the file path to extract the filename
    name = filename.split("/")
    image = cv2.imread(filename)

    imageresized = cv2.resize(image,(448,448))

    cv2.imwrite( '<destination_folder_path>/{}'.format(name[-1]), imageresized)

# read images
for (i, image_file) in enumerate(glob.iglob('<source_folder_path>/*.jpg')):
    process(image_file)
