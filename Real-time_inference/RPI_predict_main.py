import cv2
import datetime
import numpy as np
import os
from matplotlib.pyplot import imshow
from tensorflow import lite
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model


classes = {-1: "default", 0: "0_background", 1: "1_trash", 2: "2_paper", 3: "3_plastic", 4: "4_metal",
           5: "5_electronic_invoice", 6: "6_bubble_wrap", 7: "7_thin_plastic_bag", 8: "8_fruit_mesh_bag", 9: "9_thin_film_paper_cup"}


model = load_model('./Real-time_inference/V9_40ep_custom_aug.h5')


# Load the TFLite model
# tflite_model_path = os.path.abspath(os.getcwd())+"/RPI_Predict/EfficientNetB0_V8.tflite"
# TFLite_interpreter = lite.Interpreter(model_path=tflite_model_path)
# TFLite_interpreter.allocate_tensors()

# # Get input and output details
# input_details = TFLite_interpreter.get_input_details()
# output_details = TFLite_interpreter.get_output_details()

# # Assuming your model has only one input and one output (can be different for other models)
# input_shape = input_details[0]['shape']


def preprocess_image(raw_image, corners=np.array([[5, 538], [100, 120], [800, 120], [950, 538]], dtype=np.float32), size=224):

    # TODO: adjust parameters
    raw_image = cv2.resize(raw_image, (960, 540))

    target_corners = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, target_corners)
    img = cv2.warpPerspective(raw_image, matrix, (size, size))

    # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, (size, size))

    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # upload_path = os.path.join(current_dir, 'static', 'uploads', "latest.jpg")
    # cv2.imwrite(upload_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    x = np.expand_dims(img, axis=0)
    x = x.astype('float32') / 255.0
    # x = preprocess_input(x)
    # print('Input image shape:', x.shape)
    return x

def show_region(raw_image, corners= np.array([[5, 538], [100, 120], [800, 120], [950, 538]], dtype=np.float32), size= 224):
    
    img = cv2.resize(raw_image,(960,540))
    corners = np.array([[5, 538], [100, 120], [800, 120], [950, 538]], dtype=np.float32)
    contours = np.array([corners], dtype=np.int32)
    cv2.polylines(img, contours, isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.imshow('Image with Trapezoid', img)
    cv2.waitKey(1)


# capture images from camera
cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FPS,10.0)
print("FPS: {}".format(cap.get(cv2.CAP_PROP_FPS)))

# my_image = cv2.imread("C:\\Users\\ASUS\\Desktop\\HPS\\Dataset\\Dataset_Custom\\Preprocessed_Data_II\\4_metal\\00023.png")
# my_image = cv2.resize(my_image, (224, 224))

# print(my_image)

# x = np.expand_dims(my_image, axis=0)
# x = x.astype('float32') / 255.0
# # x = preprocess_input(x)
# print(x)

# output_data=model.predict(x)
# print("predicted class: ", output_data)
# Prediction = np.argmax(output_data[0])
# print("Prediction: ",classes[Prediction])

frames_list = []
while(True):
    ret, frame = cap.read()
    # cv2.imshow('frame',frame)
    show_region(frame)

    # predict
    start = datetime.datetime.now()
    input_data = preprocess_image(frame)

    output_data = model.predict(input_data)

    # # Set input data to the interpreter
    # TFLite_interpreter.set_tensor(input_details[0]['index'], input_data)

    # # Run inference
    # TFLite_interpreter.invoke()

    # # Get the output results
    # output_data = TFLite_interpreter.get_tensor(output_details[0]['index'])

    # print("predicted class: ", output_data)
    Prediction = np.argmax(output_data)
    print("Prediction: ",classes[Prediction])

    end = datetime.datetime.now()
    print("Inference time: {} ms".format(int((end-start).microseconds/1000)))
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




