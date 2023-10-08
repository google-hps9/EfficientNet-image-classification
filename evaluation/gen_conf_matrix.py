import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

classes = {-1: "default", 0: "0_background", 1: "1_trash", 2: "2_paper", 3: "3_plastic", 4: "4_metal",
           5: "5_electronic_invoice", 6: "6_bubble_wrap", 7: "7_thin_plastic_bag", 8: "8_fruit_mesh_bag", 9: "9_thin_film_paper_cup"}
classes_official = ["0_background", "1_trash", "2_paper", "3_plastic", "4_metal", "5_electronic_invoice", "6_bubble_wrap", "7_thin_plastic_bag", "8_fruit_mesh_bag", "9_thin_film_paper_cup"]

num_classes = 10
model = load_model('./model_result/V9_40ep_custom_aug.h5')

confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

def preprocess_image(img):

    x = np.expand_dims(img, axis=0)
    x = x.astype('float32') / 255.0
    # x = preprocess_input(x)
    # print('Input image shape:', x.shape)
    return x

def predict(image_path):
    input_image = cv2.imread(image_path)
    input_data = preprocess_image(input_image)
    output_data = model.predict(input_data)
    Prediction = np.argmax(output_data)
    print("Prediction: ",classes[Prediction])

    return Prediction

# root path
root_path = "C:\\Users\\ASUS\\Desktop\\HPS\\Splited_Datasets\\test_data"

for class_path in os.listdir(root_path):
    class_dir = os.path.join(root_path, class_path)
    if os.path.isdir(class_dir):
        ground_truth = classes_official.index(class_path)
        print("Ground truth: " + str(ground_truth) + " " + str(class_path))
        
        for png_image in os.listdir(class_dir):
            image_dir = os.path.join(class_dir, png_image)
            if os.path.isfile(image_dir) and png_image.endswith(".png"):
                prediction = int(predict(image_dir))
                print("Prediction: " + str(prediction) + " " + str(classes[prediction]))
                confusion_mat[ground_truth][prediction] += 1
                
print("Confusion Matrix:")
print(confusion_mat)
np.savetxt('conf_matrix_0831.txt', confusion_mat, fmt='%d', delimiter='\t')

plt.figure(figsize=(10, 8))
plt.imshow(confusion_mat, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted',fontsize=14)
plt.ylabel('Actual',fontsize=14)
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, classes_official,rotation=30)
plt.yticks(tick_marks, classes_official)
plt.show()
plt.savefig('confusion_matrix.png', format='png')
