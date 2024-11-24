import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

############################################################
######## main의 실행 결과들이 results에 저장되어있음.
######## 각 모델의 csv파일을 읽어와서 한번에 그래프 그리기
############################################################
path = "./results/*.csv"
files = glob.glob(path)

entire = []

for file in files:
    file_name = os.path.basename(file).replace(".csv", "") #use filename as key
    
    df = pd.read_csv(file)

    for i in range(len(df)):
        row = {
            'File': file_name,
            'Epochs': df['epoch'].iloc[i],
            'Train Loss': df['Training Loss'].iloc[i],
            'Valid Loss': df['Valid. Loss'].iloc[i],
            'Valid Accuracy': df['Valid. Accur.'].iloc[i],
            'Memory Usage': df['Total Memory use (MB)'].iloc[i],
            'Infer Loss': df['Infer Loss'].iloc[i],
            'Infer Accuracy': df['Infer Accuracy'].iloc[i]
        }
        entire.append(row)

entire_data = pd.DataFrame(entire)
files = entire_data['File'].unique()

colmax = int(len(files) / 2) #max number of columns
vrows = (len(files) + colmax - 1) // colmax #num of rows

# Loss
plt.figure(figsize=(12, 7))
plt.suptitle("Training & Validation Loss")

for i, file in enumerate(files):
    file_data = entire_data[entire_data['File'] == file]
    
    epochs = file_data['Epochs']
    train_loss = file_data['Train Loss']
    val_loss = file_data['Valid Loss']
    
    plt.subplot(vrows, colmax, i + 1)
    plt.plot(epochs, train_loss, 'b-o', label="Training Loss")
    plt.plot(epochs, val_loss, 'g-o', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{file}")
    plt.legend()
    plt.ylim([min(train_loss.min(), val_loss.min()), max(train_loss.max(), val_loss.max())])
    plt.tight_layout()

plt.subplots_adjust()
#plt.show()

# Validation Accuracy
plt.figure(figsize=(12, 7))
plt.suptitle("Validation Accuracy")

for i, file in enumerate(files):
    file_data = entire_data[entire_data['File'] == file]
    
    epochs = file_data['Epochs']
    val_accuracy = file_data['Valid Accuracy']
    
    plt.subplot(vrows, colmax, i + 1)
    plt.plot(epochs, val_accuracy, 'r-o', label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Valid. Accuracy")
    plt.legend()
    plt.title(f"{file}")
    plt.legend()
    plt.tight_layout()

plt.subplots_adjust()
#plt.show()

# infer result table
data = []

for file in files:
    file_data = entire_data[entire_data['File'] == file]
    
    infer_loss = file_data['Infer Loss'].values 
    infer_accuracy = file_data['Infer Accuracy'].values 
    
    if len(infer_loss) > 0 and len(infer_accuracy) > 0:
        data.append([file, round(infer_loss[-1], 3), round(infer_accuracy[-1], 3)])

columns = ['model', 'Loss', 'Accuracy']

fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')
plt.title("Inference results (rounded to the third decimal)")
table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')
table.scale(1, 1.3)
#table.auto_set_column_width([0, 1, 2])
for (i, j), cell in table.get_celld().items():
    if j == 0:
        cell.set_facecolor('#d9e2f3')
    if i == 0:
        cell.set_facecolor('#4a90e2')
        cell.set_text_props(fontsize=14, weight='bold')
    cell.set_edgecolor('black')

plt.tight_layout()
plt.show()