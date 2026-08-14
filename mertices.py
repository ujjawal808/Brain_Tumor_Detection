import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Load your ALREADY TRAINED model
# ---------------------------------------------------------
model = tf.keras.models.load_model(r"C:\Users\ujjawal baliyan\Downloads\brain_tumor_model_v2.h5")  # or .keras

# ---------------------------------------------------------
# 2. Load the Testing folder (no training happens here)
# ---------------------------------------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    r"D:\BrainTumorDetection\Brain_Tumor_Detection\archive\Testing",
    image_size=(260, 260),   # MUST match what the model was trained on
    batch_size=32,
    shuffle=False             # keep order so labels line up with predictions
)

class_names = test_ds.class_names
print("Classes found:", class_names)

# ---------------------------------------------------------
# 3. Run predictions (inference only, no fit())
# ---------------------------------------------------------
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ---------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall    = recall_score(y_true, y_pred, average='weighted')
f1        = f1_score(y_true, y_pred, average='weighted')

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nFull classification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ---------------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()