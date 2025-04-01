import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Definer sti til træningsdata
train_dir = r"C:\Users\katri\Documents\2 semester\Design og udvikling af ai systemer\King domino mini\Mini-Projekt-King-Domino\splitted_dataset\train\cropped"

# Indlæs billederne med en batch size på 32
batch_size = 32
img_height = 150
img_width = 150

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"  # Vi bruger one-hot encoding
)

# Hent label-navne (kategori-navne)
class_names = train_ds.class_names
print(f"Labels: {class_names}")

# Definer en simpel CNN-model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Antal output = antal klasser
])

# Kompiler modellen
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Træn modellen
model.fit(train_ds, epochs=10)

# --- Evaluering og confusion matrix ---
# Indlæs testbilleder
test_dir = train_dir.replace("train", "test")  # Antager testdata er i en tilsvarende mappe
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

# Få forudsigelser på testdata
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Generér confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualisering af confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
