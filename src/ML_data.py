# Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.X[index]
        y = self.y[index]

        return X, y
        

def prepare_datasets(input, output, batch_size = 100):
      # Split dataset for training 60, validation 20, and test 20%
      input_train, input_test, output_train, output_test = \
            train_test_split(input, output, test_size=0.2, random_state=len(input))

      input_train, input_validation, output_train, output_validation = \
            train_test_split(input_train, output_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


      # Create dataset
      dataset_train = Dataset(input_train, output_train)
      dataset_validation = Dataset(input_validation, output_validation)
      dataset_test = Dataset(input_test, output_test)

      # Batchsize for data
      train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True)

      validation_loader = DataLoader(dataset=dataset_validation,
                              batch_size=batch_size,
                              shuffle=True)

      test_loader = DataLoader(dataset=dataset_test,
                              batch_size=batch_size,
                              shuffle=True)

      # Print data info from training dataset (with batches)
      dataiter = iter(train_loader)
      data = dataiter.next()
      a, b = data

      print('\n--- Print from training dataset (batches) ---')
      print('conv1D takes (batches, channels = 1, features)')
      print(f'The input transmission of size = {a.shape}')
      print(f'The output envelope of size = {b.shape}')

      return train_loader, validation_loader, test_loader
