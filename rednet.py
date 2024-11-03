import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Radar3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Radar3DCNN, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Calculate the flattened size after the convolutional layers
        self.flattened_size = self._get_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
    
    def _get_flattened_size(self):
        # Helper function to compute the flattened size after convolutions
        with torch.no_grad():
            x = torch.zeros((1, 1, 100, 406, 20))  # Example input with batch size 1 and channel 1
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, self.flattened_size)  # Flattening
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Final layer for classification
        return x


num_classes = 2  # Replace with your actual number of classes
model = Radar3DCNN(num_classes)

# Check for GPU availability and move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define fake data and labels
batch_size = 10
input_shape = (1, 100, 406, 20)  # input channels, height, width, depth

# Generate random input data and labels, and move them to the GPU
fake_data = torch.randn(batch_size, *input_shape).to(device)
fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

# Timing the training loop
start_time = time.time()

# Training loop for 10 steps
num_steps = 10
for step in range(num_steps):
    # Zero gradients, perform a forward pass, compute the loss, and update weights
    optimizer.zero_grad()
    outputs = model(fake_data)
    loss = criterion(outputs, fake_labels)
    loss.backward()
    optimizer.step()
    
    # Print the loss every step for clarity
    print(f"Step [{step+1}/{num_steps}], Loss: {loss.item()}")

# Total time taken
end_time = time.time()
total_time = end_time - start_time
print(f"Total training time for {num_steps} steps: {total_time:.2f} seconds")
print(f"Average time per step: {total_time / num_steps:.2f} seconds")



#
## Initialize the model, define loss and optimizer
#num_classes = 2  # Replace with your actual number of classes
#model = Radar3DCNN(num_classes)
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#
#
#
#
#
## Define fake data and labels
#batch_size = 10
#input_shape = (1, 100, 406, 20)  # input channels, height, width, depth
#num_classes = 2
#
## Generate random input data and labels
#fake_data = torch.randn(batch_size, *input_shape)
#fake_labels = torch.randint(0, num_classes, (batch_size,))
#
## Move model and data to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = Radar3DCNN(num_classes).to(device)
#fake_data, fake_labels = fake_data.to(device), fake_labels.to(device)
#
## Define the loss and optimizer
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
## Timing the training loop
#start_time = time.time()
#
## Training loop for 10 steps
#num_steps = 10
#for step in range(num_steps):
#    # Zero gradients, perform a forward pass, compute the loss, and update weights
#    optimizer.zero_grad()
#    outputs = model(fake_data)
#    loss = criterion(outputs, fake_labels)
#    loss.backward()
#    optimizer.step()
#    
#    # Print the loss every step for clarity
#    print(f"Step [{step+1}/{num_steps}], Loss: {loss.item()}")
#
## Total time taken
#end_time = time.time()
#total_time = end_time - start_time
#print(f"Total training time for {num_steps} steps: {total_time:.2f} seconds")
#print(f"Average time per step: {total_time / num_steps:.2f} seconds")
