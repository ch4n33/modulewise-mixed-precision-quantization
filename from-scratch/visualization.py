import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("table_int_mix-8-4.csv")

epochs = df['epoch']
train_loss = df['Training Loss']
val_loss = df['Valid. Loss']
val_accuracy = df['Valid. Accur.']
train_time = df['Training Time']
val_time = df['Validation Time']

plt.figure(figsize=(12, 8))

# Training & Validation Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, 'b-o', label="Training Loss")
plt.plot(epochs, val_loss, 'g-o', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

# Validation Accuracy
plt.subplot(2, 2, 2)
plt.plot(epochs, val_accuracy, 'r-o', label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy and Training/Validation time")
plt.legend()

# Time
plt.twinx()
plt.plot(epochs, train_time, 'c-o', label="Training Time (s)")
plt.plot(epochs, val_time, 'm-o', label="Validation Time (s)")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.legend()


plt.tight_layout()
plt.show()
