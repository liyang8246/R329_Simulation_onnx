from PIL import Image
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import imagenet_classes as class_name

current_dir = os.getcwd()
label_offset = 1
outputfile = current_dir + '/output.bin'
npyoutput = np.fromfile(outputfile, dtype=np.uint8)
outputclass = npyoutput.argmax()
head5p = npyoutput.argsort()[-5:][::-1]

labelfile = current_dir + '/output_ref.bin'
npylabel = np.fromfile(labelfile, dtype=np.int8)
labelclass = npylabel.argmax()
head5t = npylabel.argsort()[-5:][::-1]

print("predict first 5 label:")
for i in head5p:
    print("    index %4d, prob %3d, name: %s"%(i, npyoutput[i], class_name.class_names[i-label_offset]))
    
print("true first 5 label:")
for i in head5t:
    print("    index %4d, prob %3d, name: %s"%(i, npylabel[i], class_name.class_names[i-label_offset]))

# Show input picture
print('Detect picture save to result.jpeg')

input_path = './input/input.bin'
npyinput = np.fromfile(input_path, dtype=np.int8)
image = np.clip(np.round(npyinput)+128, 0, 255).astype(np.uint8)
image = np.reshape(image, (224, 224, 3))
im = Image.fromarray(image)
im.save('result.jpeg')






