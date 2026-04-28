import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GESTURES = ["one", "two", "plus", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
DATASET_DIR   = "dataset"
SEQUENCE_LEN  = 30
LANDMARK_SIZE = 126
MODEL_PATH    = "signmath_model.h5"   # .h5 for TF 2.13 compatibility
# ──────────────────────────────────────────────────────────────────────────────


# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("\n" + "═" * 45)
print("  LOADING DATASET")
print("═" * 45)

X, y = [], []

for label, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATASET_DIR, gesture)
    files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])
    for f in files:
        data = np.load(os.path.join(gesture_dir, f))
        if data.shape == (SEQUENCE_LEN, LANDMARK_SIZE):
            X.append(data)
            y.append(label)
        else:
            print(f"  Skipping {gesture}/{f} — bad shape {data.shape}")

X = np.array(X)
y = np.array(y)

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
print(f"  Classes : {GESTURES}")
for i, g in enumerate(GESTURES):
    print(f"    [{i}] {g} — {np.sum(y == i)} samples")


# ─── 2. SPLIT ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train_cat = to_categorical(y_train, num_classes=len(GESTURES))
y_test_cat  = to_categorical(y_test,  num_classes=len(GESTURES))

print(f"\n  Train : {X_train.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]} samples")


# ─── 3. BUILD MODEL ───────────────────────────────────────────────────────────
print("\n" + "═" * 45)
print("  BUILDING MODEL")
print("═" * 45)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LEN, LANDMARK_SIZE)),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(len(GESTURES), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ─── 4. TRAIN ─────────────────────────────────────────────────────────────────
print("\n" + "═" * 45)
print("  TRAINING")
print("═" * 45)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    verbose=1,
)


# ─── 5. EVALUATE ──────────────────────────────────────────────────────────────
print("\n" + "═" * 45)
print("  EVALUATION")
print("═" * 45)

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n  Test accuracy : {acc * 100:.2f}%")
print(f"  Test loss     : {loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\n  Classification report:")
print(classification_report(y_test, y_pred, target_names=GESTURES))


# ─── 6. SAVE MODEL ────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\n  Model saved → {MODEL_PATH}")


# ─── 7. PLOTS ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SignMath — Training Results", fontsize=14)

# Accuracy
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
axes[2].imshow(cm, cmap="Blues")
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

n = len(GESTURES)
axes[2].set_xticks(range(n))
axes[2].set_yticks(range(n))
axes[2].set_xticklabels(GESTURES, rotation=45, ha="right", fontsize=8)
axes[2].set_yticklabels(GESTURES, fontsize=8)
for i in range(n):
    for j in range(n):
        axes[2].text(j, i, str(cm[i, j]),
                     ha="center", va="center", color="black", fontsize=9)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("\n  Plot saved → training_results.png")
print("\n  Done!\n")