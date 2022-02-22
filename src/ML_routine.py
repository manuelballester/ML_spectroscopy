# Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Structure of the Network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.l2 = nn.Linear(in_features=325, out_features=325)
        self.l3 = nn.Linear(in_features=325, out_features=325)  
        self.l4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.sigmoid(out)
        out = self.l3(out) 
        out = self.sigmoid(out)
        out = self.l4(out)
        out = self.relu(out)
        
        return out
        

def training(train_loader, validation_loader, learning_rate = 0.0001, num_epochs = 120):

      # Applying our model
      model = NeuralNet().to(device)

      # Loss and optimizer
      criterion = nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      # Training the model
      n_total_steps = len(train_loader)

      for epoch in range(num_epochs):
            for i, (X, y) in enumerate(train_loader):  

                  X = X.to(device)
                  y = y.to(device)
                  
                  # Forward pass
                  y_predicted = model(X)
                  loss = criterion(y_predicted, y)
                  
                  # Backward and optimize
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()

            for i, (X, y) in enumerate(validation_loader):  

                  X = X.to(device)
                  y = y.to(device)
                  
                  # Forward pass
                  y_predicted = model(X)
                  loss_valid = criterion(y_predicted, y)
            
            if (epoch+1)%40 == 0:
                  print(f'Epoch [{epoch+1}/{num_epochs}]')
                  print(f'Training loss: {loss.item():.5f}, validation loss: {loss_valid.item():.5f}')

      return model



def testing(wv, test_loader, model, prediction_sample):
      
      # Criterion for the total accuracy of the model
      criterion = nn.MSELoss()

      with torch.no_grad():
            list_acc = torch.tensor([1])
            
            for num, (X, y) in enumerate(test_loader):
                  X = X.to(device)
                  y = y.to(device)
                  y_predicted = model(X)
                  
                  # MSE loss and accuracy (in percentage)
                  loss = criterion(y_predicted, y).item()*1e4
                  acc = torch.tensor([100 - loss])
                  
                  # Saving accuracy in the list
                  list_acc = torch.cat((list_acc, acc), 0)

            total_acc = torch.sum(list_acc)/num
            print(f'Accuracy of the network on the test spectra: {total_acc} %')

      # To plot the values, we need to remove the batch and channel X[0,0]
      N = prediction_sample
      name = 'output/test_dataset_prediction_'+str(N)+'.png'

      plt.plot(wv, X[N,0,:], label = 'Spectrum')
      plt.plot(wv, y[N,0,:], label = 'Envelope')
      plt.plot(wv, y_predicted[N,0,:], label = 'CNN envelope')
      plt.xlabel('Wavelength (nm)')
      plt.ylabel('Transmittance')
      plt.legend()
      plt.savefig(name, dpi = 200)
      plt.show()