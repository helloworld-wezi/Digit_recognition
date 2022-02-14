import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import math
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

# def plot_images(X):
#     # if X.ndim == 1:
#     #     X = np.array([X])
#     num_images = X.shape[0]
#     num_rows = math.floor(math.sqrt(num_images))
#     num_cols = math.ceil(num_images/num_rows)
#     for i in range(num_images):
#         # reshaped_image = X[i, :].reshape(42,28)
#         plt.subplot(num_rows, num_cols, i+1)
#         plt.imshow(X[i], cmap = cm.Greys_r)
#         plt.axis('off')
#     plt.show()

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

class MLP(nn.Module):

    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        self.linear1 = nn.Linear(input_dimension, 64)
        self.linear2 = nn.Linear(64, 10)
        self.linear3 = nn.Linear(input_dimension, 64)
        self.linear4 = nn.Linear(64, 10)
        # TODO initialize model layers here


    def forward(self, x):
        xf = self.flatten(x)

        xl1 = self.linear1(xf)
        out_first_digit = self.linear2(xl1)

        xl2 = self.linear3(xf)
        out_second_digit = self.linear4(xl2)

        # TODO use model layers to predict the two digits
        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    # print(np.array(y_train)[0, 0:20])
    # print(np.array(y_train)[1, 0:20])
    # plot_images(X_train[0:20, 0])

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = MLP(input_dimension) # TODO add proper layers to MLP class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
