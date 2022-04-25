from matplotlib import pyplot as plt
from numpy.random.mtrand import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
# PyTorch libraries and modules
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, BatchNorm2d
from torch.optim import Adam
import torch.nn as nn
from ModelCreation import data_loader

shuffle = True
num_workers = 1
data_size = 13536
data_size_mock = 12


class CNN(nn.Module):
    train_loader = None
    validation_loader = None
    test_loader = None
    dataset = data_loader.AnimalsDataset()
    # train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [9001, 2267, 2267])
    train_set = torch.utils.data.random_split(dataset, [data_size])[0]

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
            Linear(12544, 10)

        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def score(model):
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(dataset=model.train_set, shuffle=False, batch_size=data_size, generator=g)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data.values()
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        predicted_labels = predicted.numpy()
        confidence_labels = conf.detach().numpy()
        for i in range(len(confidence_labels)):
            if predicted_labels[i] == 1:
                if confidence_labels[i] < 0.7:
                    predicted_labels[i] = 0
        true_labels = labels.numpy()
        print('F1: {}'.format(f1_score(y_true=true_labels, y_pred=predicted_labels, average="weighted")))
        print('Precision: {}'.format(precision_score(true_labels, predicted_labels, average="weighted")))
        print('Recall: {}'.format(recall_score(true_labels, predicted_labels, average="weighted")))
        fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, predicted_labels)
        plt.figure(1)
        plt.plot([0, 1], [0, 1])
        plt.plot(fpr, tpr, label='CNN(area={: .3f})'.format(roc_auc))
        plt.xlabel('False Positive rate')
        plt.ylabel('True Positive Rate')
        plt.title('Cage Classifier - ROC Curve')
        plt.legend(loc='best')
        plt.show()


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
    g = torch.Generator()
    g.manual_seed(42)
    model.train_loader = DataLoader(dataset=model.train_set, shuffle=shuffle, batch_size=batch_size, generator=g)
    ''' Can add validation and tests set (I used it to test the model.) '''
    # model.validation_loader = DataLoader(dataset=model.validation_set, shuffle=shuffle, batch_size=batch_size,generator=g)
    # model.test_loader = DataLoader(dataset=model.test_set, shuffle=shuffle, batch_size=batch_size,generator=g)
    return model


def train(model):
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = Adam(model.parameters(), lr=0.007)
    for epoch in range(1):  # loop over the dataset multiple times
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

    print('Finished Training')
    # save the model to disk
    torch.save(model.state_dict(), "C:\\Users\\User\\PycharmProjects\\CageClassifier\\ModelCreation\\trained_model1.pt")
    score(model)
    return model
