#infer_time = df['Infer Time']
#train_time = df['Training Time']
#val_time = df['Validation Time']

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("table_int_mix-8-4.csv")

epochs = df['epoch']
train_loss = df['Training Loss']
val_loss = df['Valid. Loss']
val_accuracy = df['Valid. Accur.']
mem = df['Total Memory use (MB)']
infer_loss = df['Infer Loss'].round(3)
infer_accuracy = df['Infer Accuracy'].round(3)


plt.figure(figsize=(9, 7))

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
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# infer result table
data = [
    ['mixedQAT', infer_loss[3], infer_accuracy[3]]
    ]
columns = ['model', 'Loss', 'Accuracy']
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')
plt.title("Inference results (rounded to the third decimal)")
table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')

plt.tight_layout()
plt.show()
