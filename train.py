import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork
from N_UTILLS import tokenize, stem, bag_of_words

# Data Preprocessing
with open('EMILE_DATA.json', 'r') as f:
    EMILE_DATA = json.load(f)

all_words = []
tags = []
xy = []

for emile in EMILE_DATA['intents']:
    tag = emile['tag']
    tags.append(tag)
    
    for pattern in emile['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!', ',', '’']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)  # Corrected the typo

# Create Dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Parameters
batch_size = 32
hidden_size = 256  # Adjusted hidden size
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001  # Adjusted learning rate
num_epochs = 3610

# DataLoader
dataset = ChatDataset()
trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training with early stopping
best_loss = float('inf')
patience = 100  # Number of epochs to wait for improvement
patience_counter = 0

for epoch in range(num_epochs):
    for (words, labels) in trainloader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.int64)  

        optimizer.zero_grad()

        outputs = model(words)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss = {loss.item():.4f}')
      
    # Early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        # Save model
        torch.save({
            "model_state": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "all_words": all_words,
            "tags": tags
        }, "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

print(f'Final Loss = {loss.item():.4f}')