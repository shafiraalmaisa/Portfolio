import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input,  Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Parameters
img_height = 244
img_width = 244
epoch = 40
lr = 0.00001

# Direktori dataset
dataset_dir = "FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET"
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(os.path.join(dataset_dir, "train"),
                                                    target_size=(img_width, img_height),
                                                    batch_size=32,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(os.path.join(dataset_dir, "test"),
                                                target_size=(img_width, img_height),
                                                batch_size=32,
                                                class_mode='categorical',
                                                shuffle=False)

# building model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compilation of the model
model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

# Pelatihan model
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=epoch)

# Evaluasi model pada dataset test
evaluation = model.evaluate(val_generator)

# Output akurasi dan loss pada dataset test
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")
print(f"Test Loss: {evaluation[0]:.4f}")

# Predict on test set
predictions = model.predict(val_generator)
predicted_classes = np.argmax(predictions, axis=1)

# True classes
true_classes = val_generator.classes

# Calculate scores
accuracy = accuracy_score(true_classes, predicted_classes)
precision_macro = precision_score(true_classes, predicted_classes, average='macro')
recall_macro = recall_score(true_classes, predicted_classes, average='macro')
f1_macro = f1_score(true_classes, predicted_classes, average='macro')
print("Accuracy:", accuracy)
print("Precision:", precision_macro)
print("Recall:",recall_macro)
print("F1-Score:", f1_macro)

# Save the model
model_save_path = 'cnn_model244.h5'
model.save(model_save_path)
print(f'Model has been saved at {model_save_path}')