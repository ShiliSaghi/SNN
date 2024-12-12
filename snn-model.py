import pickle
import snntorch as snn
import snntorch.spikeplot as splt
import snntorch.spikegen as spk
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


np.random.seed(42)

#load dataset
with open("./dataset.pkl", "rb") as f:
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

for y, x, label in train_loader:
    print(f"Batch y shape: {y.shape}")
    print(f"Batch x shape: {x.shape}")
    print(f"Batch labels shape: {label.shape}")
    break  # Just check the first batch
    

#layer parameters
Ny=32
Nx=32
num_prim_inputs = 32
num_inputs = 16
num_hidden = 1000
num_output = 1
num_steps = 80
beta = 0.95
batch_size = 128
dtype = torch.float

#define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # additional layers for y and x inputs
        self.fc_y = nn.Linear(num_inputs, Ny)
        self.fc_x = nn.Linear(num_inputs, Nx)
    
        #SNN layers
        self.fc1 = nn.Linear(Ny + Nx, num_hidden)
        self.lif1 = snn.Leaky(beta = beta)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.lif2 = snn.Leaky(beta = beta)

    def forward(self, y, x):

        # process y and x separately
        y_out= self.fc_y(y)         # Shape:[batch_size:128, num_steps:80, Ny:32]
        x_out = self.fc_x(x)        # Shape:[batch_size, num_steps, Nx] 
        # print(f"youtshape: {y_out.shape}")
        # print(f"x_outshape: {x_out.shape}")

        # concatenate y_out and x_out
        combined_input = torch.cat((y_out, x_out), dim=2)       # Shape:[batch_size:128, num_steps:80, Ny + Nx:64]
        # print(f"combined_input: {combined_input.shape}")

        #initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        #record the final layer
        mem2_rec = []
        spk2_rec = []

        for step in range(combined_input.size(1)):           # Loop over 80 time steps
            cur1 = self.fc1(combined_input[:, step, :])      # Feed one time step at a time (shape: [batch_size, 16])
            spk1, mem1 = self.lif1(cur1, mem1)
            # print(f"Shape after fc1: {cur1.shape}")
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            # print(f"Shape after fc2: {cur2.shape}")
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    
# Check for CUDA or MPS availability
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps") 
else:
    device = torch.device("cpu")

net = Net().to(device)
loss = nn.BCEWithLogitsLoss() #CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))


def print_batch_accuracy(dataY, dataX, targets, train=False):
    output, _ = net(dataY, dataX)  #[80, 128, 1]
    
    # _, idx = output.sum(dim=0).max(1) #(batch_size, num_output)/ Returns a tuple: maximum value, index of the max value along dim=1,
                                      #corresponds to the predicted class for each sample. idx shape: (batch_size,)
    # acc = np.mean((targets == idx).detach().cpu().numpy())

    pred = (output.sum(dim=0) > 0).float()  # Predicted labels: [128, 1], Sums the spike activity across the time dimension
    acc = torch.mean((pred == targets).float()).item()
    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")



#Train loop
num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0
epoch_train_loss = []  
epoch_test_loss = []  
lossMin_iter=[]
lossMax_iter=[]
lossMin_epoch=[]
lossMax_epoch=[]


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
        targets = targets.to(device)
        targets = targets.float().unsqueeze(1) # add a new dimension because the shape of targets should matche the shape of model output, ([batch_size, 1])

        # forward pass
        net.train()
        spk_rec, mem_rec = net(y, x)    # mem_rec.shape torch.Size([80, 128, 1])


        # initialize the loss & sum it
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        # for step in range(num_steps):
        #     loss_val += loss(mem_rec[step], targets) #mem_rec[step].shape = [128,1], target.shape = (128,1) 
        loss_val = loss(mem_rec[-1], targets)    
        

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


        # Store loss history for future plottin
        epoch_loss_train.append(loss_val.item())


        # Test set
        with torch.no_grad():
            net.eval()
            test_y, test_x, test_targets = next(iter(test_loader))
            test_y = test_y.to(device)
            test_x = test_x.to(device)
            test_targets = test_targets.to(device)
            test_targets = test_targets.float().unsqueeze(1)
            
            # Test set forward pass
            test_spk, test_mem = net(test_y, test_x)

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            # for step in range(num_steps):
            #     test_loss += loss(test_mem[step], test_targets)
            test_loss = loss(test_mem[-1], test_targets)  
                
            epoch_loss_test.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                loss_hist.append(loss_val.item())
                test_loss_hist.append(test_loss.item())
                
                print(f"Epoch {epoch}, Iteration {iter_counter}")
                print(f"Train Set Loss: {loss_val:.2f}")
                print(f"Test Set Loss: {test_loss:.2f}")
                print_batch_accuracy(y, x, targets, train=True)
                print_batch_accuracy(test_y, test_x, test_targets, train=False)
                print("\n")
                
            counter += 1
            iter_counter +=1
    
    print(f"Epoch {epoch}/{num_epochs}: Train Loss avg = {avg_train_loss:.4f}, Test Loss avg = {avg_test_loss:.4f}")

    print(f"Min train loss per epoch: {min(epoch_loss_train)}")
    print(f"Max train loss per epoch: {max(epoch_loss_test)}\n") 
    lossMin_epoch.append(min(epoch_loss_train))
    lossMax_epoch.append(max(epoch_loss_test))
    
    avg_train_loss = sum(epoch_loss_train) / len(epoch_loss_train)
    avg_test_loss = sum(epoch_loss_test) / len(epoch_loss_test)
    epoch_train_loss.append(avg_train_loss)
    epoch_test_loss.append(avg_test_loss)
    

    

print(f"Total min train loss: {min(lossMin_epoch)}")
print(f"Total max train loss: {max(lossMax_epoch)}")    
    



# drop_last switched to False to keep all samples
total = 0
correct = 0
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
  net.eval()
  for y_t, x_t, target_t in test_loader:
    y_t = y_t.to(device)
    x_t = x_t.to(device)
    target_t = target_t.to(device)  
    # targets = targets.to(device).float().unsqueeze(1)
    test_spk, _ = net(y_t, x_t)  # forward pass
    _, predicted = test_spk.sum(dim=0).max(1) # calculate total accuracy
    total += target_t.size(0)
    correct += (predicted == target_t).sum().item()

# print(f"Total size: {total}, Correct_Prediction: {correct}")
print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total}%")




# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
# plt.show()


# Plot Loss for Train and Test for Each Epoch
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(range(len(epoch_train_loss)), epoch_train_loss, label=' Train Loss (Epoch)')
plt.plot(range(len(epoch_test_loss)), epoch_test_loss, label='Max Train Loss (Epoch)')
plt.title("Loss Curves")
plt.legend()
plt.xlabel("Epoch / Iteration")
plt.ylabel("Loss")
plt.show()