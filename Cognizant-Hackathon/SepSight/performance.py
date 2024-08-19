import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_log(file_path):
    # Lists to store data
    epochs = []
    indices = []
    losses = []
    accuracies = []

    # Read the log file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Initialize temporary storage for current record
        current_epoch = None
        current_idx = None
        current_loss = None
        current_accuracy = None
        
        for line in lines:
            epoch_match = re.search(r'EPOCH\s*:\s*(\d+)', line)
            idx_match = re.search(r'idx\s*:\s*(\d+)', line)
            loss_match = re.search(r'loss\s*:\s*([\d.]+)', line)
            accuracy_match = re.search(r'accuracy\s*:\s*([\d.]+)', line)
            
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            if idx_match:
                current_idx = int(idx_match.group(1))
            if loss_match:
                current_loss = float(loss_match.group(1))
            if accuracy_match:
                current_accuracy = float(accuracy_match.group(1))
            
            if current_epoch is not None and current_idx is not None and current_loss is not None and current_accuracy is not None:
                epochs.append(current_epoch)
                indices.append(current_idx)
                losses.append(current_loss)
                accuracies.append(current_accuracy)
                
                # Reset for the next record
                current_epoch = None
                current_idx = None
                current_loss = None
                current_accuracy = None

    # Create DataFrame
    df = pd.DataFrame({
        'Epoch': epochs,
        'Index': indices,
        'Loss': losses,
        'Accuracy': accuracies
    })
    
    return df


def plot_accuracy(df):
    plt.figure(figsize=(10, 6))

    for epoch in df['Epoch'].unique():
        epoch_df = df[df['Epoch'] == epoch]
        plt.plot(epoch_df['Index'], epoch_df['Accuracy'], marker='o', label=f'Epoch {epoch}')

    plt.xlabel('Index')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

log_file0 = 'D:\\SEPSIGHT\\sepsight\\TrainLog0.txt'
log_file1 = 'D:\\SEPSIGHT\\sepsight\\TrainLog1.txt'
log_file2 = 'D:\\SEPSIGHT\\sepsight\\TrainLog2.txt'
df0 = parse_log(log_file0)
df1 = parse_log(log_file1)
df2 = parse_log(log_file2)
combined_df = pd.concat([df0, df1, df2], ignore_index=True)
plot_accuracy(combined_df)