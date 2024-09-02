# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
# from tensorflow.keras.models import load_model
# import tensorflow as tf
# import cv2
# from matplotlib import pyplot as plt
# import numpy as np
# import imghdr
# import os

# data_dir = 'data'
# image_exts = ['jpeg', 'jpg', 'png']

# # Validating and Cleaning Image Data
# for image_class in os.listdir(data_dir):
#     class_path = os.path.join(data_dir, image_class)
#     for image in os.listdir(class_path):
#         image_path = os.path.join(class_path, image)
#         try:
#             tip = imghdr.what(image_path)
#             if tip not in image_exts:
#                 print(f'Removing file {image_path} due to unacceptable type: {tip}')
#                 os.remove(image_path)
#         except Exception as e:
#             print(f'Error processing image {image_path}: {e}')



# # Creating a Data Pipeline
# data = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     batch_size=32,  # Customize batch size
#     image_size=(256, 256),  # Resize images to a standard size
#     shuffle=True  # Shuffle the data for better training
# )

# # Store class names before mapping
# class_names = data.class_names

# # Plotting Images with Labels
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# data_iterator = data.as_numpy_iterator()
# batch = data_iterator.next()

# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(class_names[np.argmax(batch[1][idx])])
#     # ax[idx].title.set_text(class_names[batch[1][idx]])
# plt.show()

# # Data Preprocessing: Scaling (0-1)
# data = data.map(lambda x, y: (x / 255.0, y))
# scaled_iterator = data.as_numpy_iterator()
# scaled_batch = scaled_iterator.next()

# print(scaled_batch[0].min())  # Expected to be 0.0
# print(scaled_batch[0].max())  # Expected to be 1.0

# # Visualizing After Scaling
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx, img in enumerate(scaled_batch[0][:4]):
#     ax[idx].imshow(img)
#     ax[idx].title.set_text(class_names[scaled_batch[1][idx].argmax()])
# plt.show()

# # Data Split(training, validating and testing)
# train_size = int(len(data)*.7)
# val_size = int(len(data)*.2)+1
# test_size = int(len(data)*.1)+1

# train = data.take(train_size)
# val = data.skip(train_size).take(val_size)
# test = data.skip(train_size+val_size).take(test_size)

# # BUILDING MODEL
# model = Sequential([
#     Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
#     MaxPooling2D(),
#     Conv2D(32, (3, 3), 1, activation='relu'),
#     MaxPooling2D(),
#     Conv2D(16, (3, 3), 1, activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])


# # compilation with adam optimiser
# model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])

# # Summary
# print(model.summary())

# # Training Model:
# logdir= 'logs'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# hist = model.fit(train,epochs=20,validation_data=val,callbacks=[tensorboard_callback])
# # print("hist = ",hist.history)

# # Visualizing LOSS
# fig= plt.figure()
# plt.plot(hist.history['loss'],color='teal',label='loss')
# plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
# fig.suptitle('Loss',fontsize=30)
# plt.legend(loc='upper left')
# plt.show()

# # Visualizing ACCURACY
# fig = plt.figure()
# plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
# plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
# fig.suptitle('Accuracy', fontsize=30)
# plt.legend(loc='upper left')
# plt.show()

# # Evaluating Performance
# pre = Precision()
# re = Recall()
# bAcc = BinaryAccuracy()

# # Using test dataset for evaluation
# for batch in test.as_numpy_iterator():
#     X, y = batch
#     yhat = model.predict(X)
#     pre.update_state(y, yhat)
#     re.update_state(y, yhat)
#     bAcc.update_state(y, yhat)


# print(f"Precision: {pre.result().numpy()}")
# print(f"Recall: {re.result().numpy()}")
# print(f"Accuracy: {bAcc.result().numpy()}")

# # Prediction function
# def final(yhat):
#     if yhat > 0.5:
#         print(f"Predicted class is SAD")
#     else:
#         print(f"Predicted class is HAPPY")

# # Testing :
# happyImg = cv2.imread('happyTest.jpeg')
# resized_happyImg = tf.image.resize(happyImg, (256, 256))
# plt.imshow(resized_happyImg.numpy().astype(int))
# plt.show()

# yhat = model.predict(np.expand_dims(resized_happyImg / 255.0, 0))
# final(yhat[0][0])


# # sadImage Testing
# sadImg = cv2.imread('sadTest.jpeg')
# resized_sadImg = tf.image.resize(sadImg, (256, 256))
# plt.imshow(resized_sadImg.numpy().astype(int))
# plt.show()

# yhat = model.predict(np.expand_dims(resized_sadImg / 255.0, 0))
# final(yhat[0][0])


# # Model Saving
# model.save(os.path.join('model','Emotion_Predictor.h5'))
# new_model = load_model(os.path.join('model','Emotion_Predictor.h5'))
# print("Saved Model Prediction :")
# yhatnew = new_model.predict(np.expand_dims(resized_sadImg/255.0,0))

# final(yhatnew[0][0])


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imghdr
import os

data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'png']

# Validating and Cleaning Image Data
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        try:
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print(f'Removing file {image_path} due to unacceptable type: {tip}')
                os.remove(image_path)
        except Exception as e:
            print(f'Error processing image {image_path}: {e}')

# Data Augmentation
data_gen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,  # Split data into training and validation sets
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

# Creating Data Pipelines
train_data = data_gen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Store class names
class_names = list(train_data.class_indices.keys())

# Model Definition
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Summary
print(model.summary())

# Training the Model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])

# Visualizing Training Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(hist.history['loss'], label='Training Loss')
ax1.plot(hist.history['val_loss'], label='Validation Loss')
ax1.legend(loc='upper right')
ax1.set_title('Loss')

ax2.plot(hist.history['accuracy'], label='Training Accuracy')
ax2.plot(hist.history['val_accuracy'], label='Validation Accuracy')
ax2.legend(loc='lower right')
ax2.set_title('Accuracy')

plt.show()

# Evaluating Model Performance on Validation Set
val_loss, val_acc, val_prec, val_recall = model.evaluate(val_data)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")
print(f"Validation Precision: {val_prec}")
print(f"Validation Recall: {val_recall}")

# Model Saving
model_save_path = 'model/Emotion_Predictor.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

# Loading and Testing the Model
new_model = load_model(model_save_path)
print("Model loaded successfully.")

#Prediction Function
def final_prediction(img_path):
    img = cv2.imread(img_path)
    resized_img = tf.image.resize(img, (256, 256))
    plt.imshow(resized_img.numpy().astype(int))
    plt.show()
    yhat = new_model.predict(np.expand_dims(resized_img / 255.0, 0))[0][0]
    if yhat > 0.5:
        print("Predicted class is SAD")
    else:
        print("Predicted class is HAPPY")

# Testing with Images
final_prediction('happyTest.jpeg')
final_prediction('sadTest.jpeg')











