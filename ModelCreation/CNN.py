# for evaluating the model

from numpy.random.mtrand import shuffle
from torch.utils.data import DataLoader
import torch.nn.functional as F
# PyTorch libraries and modules
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, BatchNorm2d
from torch.optim import Adam
import torch.nn as nn
import data_loader
import numpy as np

# CURRENTLY 2016 IMAGES
# learning_rate = 0.00001
# batch_size = int(2016 / 8)
shuffle = False
pin_memory = True
num_workers = 1
data_size = 13535

class CNN(nn.Module):
    train_loader = None
    validation_loader = None
    test_loader = None
    dataset = data_loader.AnimalsDataset()
    train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [9001, 2267, 2267])
    # train_set = torch.utils.data.random_split(dataset, [data_size])

    def __init__(self):
        super(CNN, self).__init__()
        self.train_CNN = True
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            # Linear(4 * 7 * 7 * 100, 10)
            Linear(12544, 10)

        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def get_train_accuracy(model):
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(dataset=model.train_set, shuffle=shuffle, batch_size=2267,generator=g)
    for i, data in enumerate(train_loader, 0):
        accuracy = single_predict(data, model)
        return accuracy


def get_validation_accuracy(model):
    g = torch.Generator()
    g.manual_seed(42)
    validation_loader = DataLoader(dataset=model.validation_set, shuffle=shuffle, batch_size=2267,generator=g)
    for i, data in enumerate(validation_loader, 0):
        print("VAL ONLY ONCE.")
        accuracy = single_predict(data, model)
        return accuracy


def get_test_accuracy(model):
    g = torch.Generator()
    g.manual_seed(42)
    test_loader = DataLoader(dataset=model.test_set, shuffle=False, batch_size=2267,generator=g)
    df = model.test_loader.dataset.dataset.data
    lst = model.test_loader.dataset.indices
    # TODO put df only on lst's idices- will make it 225
    for i, data in enumerate(test_loader, 0):
        accuracy, conf, predictions = single_predict(data, model)
        # df["Confidence Level"] = conf
        # df["Predictions"] = predictions
        # filepath = os.path.join("animals_images")
        # df.to_csv(filepath + "/out.csv")
        return accuracy


def single_predict(data, model):
    inputs, labels = data.values()
    outputs = model(inputs)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    conf, predicted = torch.max(probs, 1)
    first = predicted.numpy()
    second = labels.numpy()
    counter = np.sum(first == second)
    accuracy = ((counter / 2267) * 100)
    return accuracy, conf, first


def define_parameters(batch_size):
    # defining the model
    model = CNN()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=0.07)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    # validation_set
    g = torch.Generator()
    g.manual_seed(42)
    model.train_loader = DataLoader(dataset=model.train_set, shuffle=shuffle, batch_size=batch_size,generator=g)
    model.validation_loader = DataLoader(dataset=model.validation_set, shuffle=shuffle, batch_size=batch_size,generator=g)
    model.test_loader = DataLoader(dataset=model.test_set, shuffle=shuffle, batch_size=batch_size,generator=g)
    return model


def train(model):
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = Adam(model.parameters(), lr=0.007)
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(model.train_loader, 0):
            # get the inputs
            inputs, labels = data.values()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics

    print('Finished Training')
    # save the model to disk
    # filename = 'trained_cnn_cage_classifier.sav'
    # pickle.dump(model, open(filename, 'wb'))

    # some time later...

    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    return model
