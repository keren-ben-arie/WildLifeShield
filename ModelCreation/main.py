import numpy as np
import torch
from matplotlib import pyplot as plt
import CNN


def plot_model(batch_sizes, train_results, validation_results, test_results):
    # plotting the training and validation loss
    plt.plot()
    plt.plot(batch_sizes, train_results, color='pink', label="Train")
    plt.plot(batch_sizes, validation_results, color='purple', label="Validation")
    plt.plot(batch_sizes, test_results, color='blue', label='Test')
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Free/Caged Animals Classifier \n Accuracy Rates as a function of Batch Size")
    plt.show()


if __name__ == '__main__':
    # test_results = []
    # validation_results = []
    # train_results = []
    # batch_sizes = [150, 250, 350]
    model = CNN.define_parameters(250)
    print("Starting Training.")
    model = CNN.train(model)
    print("Model Saved.")

    # print("done")

    # for batch in batch_sizes:
    #     model = CNN.define_parameters(batch)
    #     model = CNN.train(model)
    #     train_results.append(CNN.get_train_accuracy(model))
    #     validation_results.append(CNN.get_validation_accuracy(model))
    #     test_results.append(CNN.get_test_accuracy(model))
    # plot_model(np.array(batch_sizes), np.array(train_results), np.array(validation_results), np.array(test_results))
