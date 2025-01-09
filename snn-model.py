import pickle
import snntorch as snn
import snntorch.spikeplot as splt
import snntorch.spikegen as spk
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from snntorch import functional as SF

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

np.random.seed(42)

#load dataset
with open("/content/drive/MyDrive/Colab Notebooks/SNN/Dataset/dataset.pkl", "rb") as f:
    data = pickle.load(f)

#split data into test and train
indexes = np.random.permutation(len(data))
train_size = int(0.8 * len(data))
train_indexes = indexes[:train_size]
test_indexes = indexes[train_size:]

train_data = [data[i] for i in train_indexes]
test_data = [data[i] for i in test_indexes]

print(f"training data size {len(train_data)}")
print(f"test data size {len(test_data)}")

#create CustomDataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.y = []
        self.x = []
        self.labels = []

        #process each sample
        for sample in data:
            y = sample[0]
            x = sample[1]
            label = sample[2]

            self.y.append(y)
            self.x.append(x)
            self.labels.append(label)

        self.y = torch.tensor(self.y, dtype = torch.float32)
        self.x = torch.tensor(self.x, dtype= torch.float32)
        self.labels = torch.tensor(self.labels, dtype= torch.long)

    def __len__(self):
        return(len(self.x))

    def __getitem__(self, index):
        return self.y[index], self.x[index], self.labels[index]

#create train and test dataset
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

#create dataloader
train_loader = DataLoader(train_dataset, batch_size=128 , shuffle=True, drop_last= True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle= True, drop_last= True)


import pandas as pd
print("Class distribution in the original training dataset:", pd.Series(train_dataset.labels).value_counts())
print("Class distribution in the original testing dataset:", pd.Series(test_dataset.labels).value_counts())

for y, x, label in train_loader:
    print(f"Batch y shape: {y.shape}")
    print(f"Batch x shape: {x.shape}")
    print(f"Batch labels shape: {label.shape}")
    break  # Just check the first batch




#layer parameters
Ny=32
Nx=32
num_inputs = 16
num_hidden = 256
num_output = 1
num_steps = 80
beta = 0.95
batch_size = 128
dtype = torch.float

#define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize additional layers for y and x inputs
        self.fc_y = nn.Linear(num_inputs, Ny)
        self.lif_y = snn.Leaky(beta = beta)
        self.fc_x = nn.Linear(num_inputs, Nx)
        self.lif_x = snn.Leaky(beta = beta)

        # initialize SNN layers
        self.fc1 = nn.Linear(Ny + Nx, num_hidden)
        self.lif1 = snn.Leaky(beta = beta)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.lif2 = snn.Leaky(beta = beta)

    def forward(self, y, x):

        # initialize hidden states at t=0
        mem_y = self.lif_y.init_leaky()
        mem_x = self.lif_x.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem2_rec = []
        spk2_rec = []

        for step in range(num_steps):                         #Loop over 80 time steps
            cur_y = self.fc_y(y[step])
            spk_y, mem_y = self.lif_y(cur_y, mem_y)
            cur_x = self.fc_x(x[step])
            spk_x, mem_x = self.lif_x(cur_x, mem_x)

            combined_input = torch.cat((spk_y, spk_x), dim=1) #Shape:[batch_size:128, Ny + Nx:64]

            cur1 = self.fc1(combined_input)
            spk1, mem1 = self.lif1(cur1, mem1)                #torch.Size([128, 256])
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)                #torch.Size([128, 1])
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)  #[num_steps:80, batch_size:128, 1]

# Load the network onto CUDA if available
net = Net().to(device)



def print_batch_accuracy(dataY, dataX, targets, train=False):
    output, _ = net(dataY, dataX)  #[80, 128, 1]
pred = (output.sum(dim=0) > num_steps / 2).float()
    acc = (pred == targets).float().mean().item()


    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

    return acc


loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))



#Train loop
num_epochs = 5
loss_hist = []
test_loss_hist = []
counter = 0
epoch_train_loss = []
epoch_test_loss = []
# lossMin_iter=[]
# lossMax_iter=[]
lossMin_epoch=[]
lossMax_epoch=[]
train_acc_hist = []
test_acc_hist = []


# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader) #len:625
    #print("train_batch:", len(train_batch))
    epoch_loss_train = []
    epoch_loss_test = []
    avg_train_loss =0
    avg_test_loss =0

    # Minibatch training loop
    for y, x, targets in train_batch:
        y = y.to(device)
        x = x.to(device)
        y = y.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        targets = targets.to(device)
        targets2 = targets.float().unsqueeze(1) # add a new dimension because the shape of targets should matche the shape of model output, ([batch_size, 1])

        # forward pass
        net.train()
        spk_rec, mem_rec = net(y, x)    # mem_rec.shape torch.Size([80, 128, 1])
        # print(f"Shape spk.shape:{spk_rec.shape}, mem.shape:{mem_rec.shape}")


        # initialize the loss & sum it
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets2) #mem_rec[step].shape = [128,1], target.shape = (128,1)
            
        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


        # Store loss history for future plotting
        epoch_loss_train.append(loss_val.item())


        # Test set
        with torch.no_grad():
            net.eval()
            test_y, test_x, test_targets = next(iter(test_loader))
            test_y = test_y.to(device)
            test_x = test_x.to(device)
            test_y = test_y.permute(1, 0, 2)
            test_x = test_x.permute(1, 0, 2)
            test_targets = test_targets.to(device)
            test_targets2 = test_targets.float().unsqueeze(1)

            # Test set forward pass
            test_spk, test_mem = net(test_y, test_x)

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
              test_loss += loss(test_mem[step], test_targets2)
            # test_loss = loss(test_mem[-1], test_targets2)

            epoch_loss_test.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                loss_hist.append(loss_val.item())
                test_loss_hist.append(test_loss.item())

                print(f"Epoch {epoch+1}, Iteration {iter_counter}")
                print(f"Train Set Loss: {loss_val}")
                print(f"Test Set Loss: {test_loss}")
                train_acc = print_batch_accuracy(y, x, targets2, train=True)
                test_acc = print_batch_accuracy(test_y, test_x, test_targets2, train=False)
                print("\n")

                train_acc_hist.append(train_acc)
                test_acc_hist.append(test_acc)


            counter += 1
            iter_counter +=1


    print(f"\nMin train loss per epoch: {min(epoch_loss_train)}")
    # print(f"Max test loss per epoch: {max(epoch_loss_test)}")
    lossMin_epoch.append(min(epoch_loss_train))
    lossMax_epoch.append(max(epoch_loss_test))

    avg_train_loss = sum(epoch_loss_train) / len(epoch_loss_train)
    avg_test_loss = sum(epoch_loss_test) / len(epoch_loss_test)
    epoch_train_loss.append(avg_train_loss)
    epoch_test_loss.append(avg_test_loss)
    print(f"Epoch {(epoch+1)}/{num_epochs}: Train Loss avg = {avg_train_loss:.4f}, Test Loss avg = {avg_test_loss:.4f}\n")



print(f"epoch {lossMin_epoch.index(min(lossMin_epoch))},  Total min train loss: {min(lossMin_epoch)}")
# print(f"Total max train loss: {max(lossMax_epoch)}")


# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Plot Loss for Train and Test for Each Epoch
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(range(len(epoch_train_loss)), epoch_train_loss, label='Train Loss (epoch)')
plt.plot(range(len(epoch_test_loss)), epoch_test_loss, label='Test Loss (epoch)')
plt.title("Loss Curves")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot acc
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(train_acc_hist, label='Train acc (epoch)')
plt.plot(test_acc_hist, label='test acc (epoch)')
plt.title("Accuracy Curves")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


#Test whole testdata, drop_last switched to False to keep all samples
total = 0
correct = 0
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
total_loss=0

with torch.no_grad():
  net.eval()
  for y_t, x_t, target_t in test_loader:
    y_t = y_t.to(device)
    x_t = x_t.to(device)
    y_t = y_t.permute(1, 0, 2)
    x_t = x_t.permute(1, 0, 2)
    target_t = target_t.to(device)
    targets = target_t.to(device).float().unsqueeze(1) # Shape adjustment for BCE loss
    test_spk, test_mem = net(y_t, x_t)  # forward pass

    # Calculate loss
    batch_loss = torch.zeros((1), dtype=dtype, device=device)
    for step in range(num_steps):
      batch_loss += loss(test_mem[step], targets)
    total_loss += batch_loss.item()

    # Calculate predictions and accuracy
    # pred = (output.sum(dim=0) > num_steps / 2).float()
    # acc = (pred == targets).float().mean().item()

    predicted = (test_spk.sum(dim=0) > num_steps / 2).float() 
    correct += (predicted == targets).float().sum().item()

    total += targets.size(0)

print(f"Total size: {total}, Correct_Prediction: {correct}")
print(f"Test Set Accuracy: {100 * correct / total}%")
print(f"Average Test Loss: {total_loss / len(test_loader):.4f}")
