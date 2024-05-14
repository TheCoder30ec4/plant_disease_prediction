import tensorflow as tf
from tensorflow.keras import models, layers
import  matplotlib.pyplot as plt
from IPython.display import HTML
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

IMAGE_SIZE = 256
CHANNELS =3

train_datagen = ImageDataGenerator(
	rescale = 1./255,
	horizontal_flip =True,
	rotation_range = 10
)
train_generator = train_datagen.flow_from_directory(
	"C:/Users/Varun/OneDrive/Desktop/ml project/Data_Set/dataset/train",
	target_size=(IMAGE_SIZE,IMAGE_SIZE),
	batch_size = 32,
	class_mode='sparse'
)
for image_batch, label_batch in train_generator:
	print(image_batch[0])
	break
validation_datagen = ImageDataGenerator(
	rescale = 1./255,
	rotation_range = 10,
	horizontal_flip =True
)
validation_generator = validation_datagen.flow_from_directory(
	"C:/Users/Varun/OneDrive/Desktop/ml project/Data_Set/dataset/val",
	target_size=(IMAGE_SIZE,IMAGE_SIZE),
	batch_size = 32,
	class_mode='sparse'
)
test_datagen = ImageDataGenerator(
	rescale = 1./255,
	rotation_range = 10,
	horizontal_flip =True
)
test_generator = validation_datagen.flow_from_directory(
	"C:/Users/Varun/OneDrive/Desktop/ml project/Data_Set/dataset/test",
	target_size=(IMAGE_SIZE,IMAGE_SIZE),
	batch_size = 32,
	class_mode='sparse'
)
input_shape = (IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes = 15
model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

print(model.summary())

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
history = model.fit(
	train_generator,
	steps_per_epoch=452,
	batch_size= 32,
	validation_data = validation_generator,
	validation_steps =452,
	verbose=1,
	epochs=50
)
scores = model.evaluate(test_generator)
print(scores)

print(history)
print(history.params)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
print(val_acc)
model.save("../model_1.h5")

EPOCHS = 50

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

for image_batch, label_batch in test_generator:
    first_image = image_batch[0]
    first_label = int(label_batch[0])   
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])
    
    break
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i])
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

plt.figure(figsize=(15, 15))
for images, label in test_generator:
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        
        predicted_class, confidence = predict(model, images[i])
        actual_class = class_names[int(label[i])] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")
    break


