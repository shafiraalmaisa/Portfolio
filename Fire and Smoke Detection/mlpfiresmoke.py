import numpy as np
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Parameters
img_height = 299
img_width = 299
epoch = 40
lr = 0.0001

# Direktori dataset
dataset_dir = "FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET"
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(os.path.join(dataset_dir, "train"),
                                                    target_size=(img_height, img_width),
                                                    batch_size=32,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(os.path.join(dataset_dir, "test"),
                                                target_size=(img_height, img_width),
                                                batch_size=32,
                                                class_mode='categorical',
                                                shuffle=False)


# Load Xception model
extraction_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3), pooling='avg')

# Freeze the base_model
extraction_model.trainable = False

# Add custom top layers
model = Sequential([
    extraction_model,
    Dense(244, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

# model.summary()

# Train the model
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=epoch)

# Evaluate model
evaluation = model.evaluate(val_generator)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")
print(f"Test Loss: {evaluation[0]:.4f}")

# Predict on test set
predictions = model.predict(val_generator)
predicted_classes = np.argmax(predictions, axis=1)

# True classes
true_classes = test_ds.classes

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
model_save_path = 'mlp_model299.h5'
model.save(model_save_path)
print(f'Model has been saved at {model_save_path}')