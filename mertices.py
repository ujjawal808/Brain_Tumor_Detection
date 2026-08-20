import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize


# =========================================================
# 1. LOAD ALREADY TRAINED MODEL
# =========================================================

MODEL_PATH = r"C:\Users\ujjawal baliyan\Downloads\brain_tumor_model_v2.h5"

model = tf.keras.models.load_model(MODEL_PATH)

print("==========================================")
print("Model loaded successfully")
print("==========================================")


# =========================================================
# 2. LOAD TESTING DATASET
# =========================================================

TEST_DIR = r"D:\BrainTumorDetection\Brain_Tumor_Detection\archive\Testing"

IMG_SIZE = (260, 260)
BATCH_SIZE = 32

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)

print("\nClasses found:")
for i, class_name in enumerate(class_names):
    print(i, ":", class_name)

print("\nNumber of classes:", num_classes)


# =========================================================
# 3. RUN MODEL PREDICTIONS
# =========================================================

y_true = []
y_pred = []
y_prob = []

print("\n==========================================")
print("Running predictions...")
print("==========================================")

for images, labels in test_ds:

    # Get probability predictions
    predictions = model.predict(
        images,
        verbose=0
    )

    # Store actual labels
    y_true.extend(
        labels.numpy()
    )

    # Store predicted class
    y_pred.extend(
        np.argmax(predictions, axis=1)
    )

    # Store probability for every class
    y_prob.extend(
        predictions
    )


# Convert lists to NumPy arrays

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)


print("\nTotal test images:", len(y_true))
print("Prediction shape:", y_prob.shape)


# =========================================================
# 4. ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


# =========================================================
# 5. PRECISION
# =========================================================

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# =========================================================
# 6. RECALL
# =========================================================

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# =========================================================
# 7. F1 SCORE
# =========================================================

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# =========================================================
# 8. PRINT OVERALL RESULTS
# =========================================================

print("\n==========================================")
print("OVERALL TEST PERFORMANCE")
print("==========================================")

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")


# =========================================================
# 9. CLASSIFICATION REPORT
# =========================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# =========================================================
# 10. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_true, y_pred)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")
print(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - EfficientNetV2S")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

# IMPORTANT:
# Do NOT use plt.show() here
plt.close()


# =========================================================
# 11. ROC-AUC
# =========================================================

print("\n==========================================")
print("CALCULATING ROC-AUC...")
print("==========================================")


# Binarize true labels
y_true_bin = label_binarize(
    y_true,
    classes=np.arange(num_classes)
)

print("Binarized labels shape:", y_true_bin.shape)
print("Prediction probabilities shape:", y_prob.shape)


# =========================================================
# 12. CALCULATE ROC-AUC FOR EACH CLASS
# =========================================================

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):

    fpr[i], tpr[i], _ = roc_curve(
        y_true_bin[:, i],
        y_prob[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )


# =========================================================
# 13. PRINT CLASS-WISE ROC-AUC
# =========================================================

print("\n==========================================")
print("CLASS-WISE ROC-AUC")
print("==========================================")

for i, class_name in enumerate(class_names):

    print(
        f"{class_name:<20}: {roc_auc[i]:.4f}"
    )


# =========================================================
# 14. MACRO AVERAGE ROC-AUC
# =========================================================

macro_auc = roc_auc_score(
    y_true_bin,
    y_prob,
    multi_class="ovr",
    average="macro"
)

print("\n==========================================")
print(f"MACRO-AVERAGE ROC-AUC: {macro_auc:.4f}")
print("==========================================")


# =========================================================
# 15. PLOT ROC-AUC CURVE
# =========================================================

plt.figure(figsize=(9, 7))

for i, class_name in enumerate(class_names):

    plt.plot(
        fpr[i],
        tpr[i],
        linewidth=2,
        label=f"{class_name} (AUC = {roc_auc[i]:.4f})"
    )


# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate",
    fontsize=12
)

plt.ylabel(
    "True Positive Rate",
    fontsize=12
)

plt.title(
    "ROC-AUC Curve - EfficientNetV2S",
    fontsize=14
)

plt.legend(
    loc="lower right"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "roc_auc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

# Now show the ROC plot
plt.show()

plt.close()


# =========================================================
# 16. FINAL RESULTS
# =========================================================

print("\n==========================================")
print("FINAL MODEL EVALUATION")
print("==========================================")

print(f"Accuracy       : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1 Score       : {f1 * 100:.2f}%")
print(f"Macro ROC-AUC  : {macro_auc:.4f}")

print("\nClass-wise ROC-AUC:")

for i, class_name in enumerate(class_names):

    print(
        f"{class_name:<20}: {roc_auc[i]:.4f}"
    )

print("\n==========================================")
print("Files saved:")
print("1. confusion_matrix.png")
print("2. roc_auc_curve.png")
print("==========================================")

# =========================================================
# 11. BINARIZE TRUE LABELS FOR ROC-AUC
# =========================================================

y_true_bin = label_binarize(
    y_true,
    classes=np.arange(num_classes)
)

print("\nBinarized labels shape:", y_true_bin.shape)


# =========================================================
# 12. CALCULATE ROC CURVE AND AUC FOR EACH CLASS
# =========================================================

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):

    fpr[i], tpr[i], thresholds = roc_curve(
        y_true_bin[:, i],
        y_prob[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )


# =========================================================
# 13. PRINT CLASS-WISE ROC-AUC
# =========================================================

print("\n==========================================")
print("CLASS-WISE ROC-AUC")
print("==========================================")

for i, class_name in enumerate(class_names):

    print(
        f"{class_name:<20} : "
        f"{roc_auc[i]:.4f}"
    )


# =========================================================
# 14. MACRO-AVERAGE ROC-AUC
# =========================================================

macro_auc = roc_auc_score(
    y_true_bin,
    y_prob,
    multi_class="ovr",
    average="macro"
)

print("\n==========================================")
print(f"Macro-Average ROC-AUC : {macro_auc:.4f}")
print("==========================================")


# =========================================================
# 15. PLOT ROC-AUC CURVES
# =========================================================

plt.figure(figsize=(9, 7))

for i, class_name in enumerate(class_names):

    plt.plot(
        fpr[i],
        tpr[i],
        linewidth=2,
        label=f"{class_name} (AUC = {roc_auc[i]:.4f})"
    )


# Random classifier reference line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate",
    fontsize=12
)

plt.ylabel(
    "True Positive Rate",
    fontsize=12
)

plt.title(
    "ROC-AUC Curve - EfficientNetV2S",
    fontsize=14
)

plt.legend(
    loc="lower right"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# Save ROC curve

plt.savefig(
    "roc_auc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 16. FINAL SUMMARY
# =========================================================

print("\n==========================================")
print("FINAL MODEL EVALUATION")
print("==========================================")

print(f"Accuracy              : {accuracy * 100:.2f}%")
print(f"Precision             : {precision * 100:.2f}%")
print(f"Recall                : {recall * 100:.2f}%")
print(f"F1 Score              : {f1 * 100:.2f}%")
print(f"Macro ROC-AUC         : {macro_auc:.4f}")

print("\nClass-wise ROC-AUC:")

for i, class_name in enumerate(class_names):

    print(
        f"  {class_name:<18}: "
        f"{roc_auc[i]:.4f}"
    )

print("\n==========================================")
print("Files saved:")
print("1. confusion_matrix.png")
print("2. roc_auc_curve.png")
print("==========================================")