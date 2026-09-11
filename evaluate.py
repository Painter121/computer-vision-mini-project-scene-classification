import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 16

# ========================= Load Model =========================
model = tf.keras.models.load_model("scene_model.keras")

# ========================= Load Training History ====================
with open("train_history.json", "r") as f:
    history = json.load(f)

# ========================= Prepare Test Data =========================
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_gen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ========================= Evaluate Model ========================
loss, acc = model.evaluate(test_data)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# ========================= Prediction =========================
pred = model.predict(test_data)
y_pred = np.argmax(pred, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

print("\nClassification Report\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ========================= Confusion Matrix ================
cm = confusion_matrix(y_true, y_pred)

# ========================= Accuracy & Loss ===================
plt.figure(figsize=(14, 5))

# ----- Accuracy -----
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"], loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# ----- Loss -----
plt.subplot(1, 2, 2)
plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ========================= Confusion Matrix =========================
plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Greens",
    square=True,
    cbar=True,
    annot_kws={"size": 11}
)

plt.title("Confusion Matrix", fontsize=16, pad=15)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.show()