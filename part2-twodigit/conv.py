import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.first_01 = nn.Conv2d(1, 2, (3, 3))
        self.first_02 = nn.ReLU()
        self.first_03 = nn.Conv2d(2, 32, (5, 5))
        self.first_04 = nn.ReLU()
        self.first_05 = nn.Conv2d(32, 32, (5, 5))
        self.first_06 = nn.BatchNorm2d(32)
        self.first_07 = nn.ReLU()
        self.first_08 = nn.MaxPool2d((2, 2))
        self.first_09 = nn.Dropout(0.25)
        self.first_10 = nn.Conv2d(32, 64, (3, 3))
        self.first_11 = nn.ReLU()
        self.first_12 = nn.Conv2d(64, 64, (3, 3))
        self.first_13 = nn.BatchNorm2d(64)
        self.first_14 = nn.ReLU()
        self.first_15 = nn.MaxPool2d((2, 2))
        self.first_16 = nn.Dropout(0.25)
        self.first_17 = nn.Flatten()
        self.first_18 = nn.ReLU()
        self.first_19 = nn.Linear(768, 256)
        self.first_20 = nn.ReLU()
        self.first_21 = nn.Linear(256, 64)
        self.first_22 = nn.ReLU()
        self.first_23 = nn.Dropout(0.25)
        self.first_24 = nn.Linear(64, 10)

        self.second_01 = nn.Conv2d(1, 2, (3, 3))
        self.second_02 = nn.ReLU()
        self.second_03 = nn.Conv2d(2, 32, (5, 5))
        self.second_04 = nn.ReLU()
        self.second_05 = nn.Conv2d(32, 32, (5, 5))
        self.second_06 = nn.BatchNorm2d(32)
        self.second_07 = nn.ReLU()
        self.second_08 = nn.MaxPool2d((2, 2))
        self.second_09 = nn.Dropout(0.25)
        self.second_10 = nn.Conv2d(32, 64, (3, 3))
        self.second_11 = nn.ReLU()
        self.second_12 = nn.Conv2d(64, 64, (3, 3))
        self.second_13 = nn.BatchNorm2d(64)
        self.second_14 = nn.ReLU()
        self.second_15 = nn.MaxPool2d((2, 2))
        self.second_16 = nn.Dropout(0.25)
        self.second_17 = nn.Flatten()
        self.second_18 = nn.ReLU()
        self.second_19 = nn.Linear(768, 256)
        self.second_20 = nn.ReLU()
        self.second_21 = nn.Linear(256, 64)
        self.second_22 = nn.ReLU()
        self.second_23 = nn.Dropout(0.25)
        self.second_24 = nn.Linear(64, 10)
        # TODO initialize model layers here

    def forward(self, x):
        out_first_digit = self.first_01(x)
        out_first_digit = self.first_02(out_first_digit)
        out_first_digit = self.first_03(out_first_digit)
        out_first_digit = self.first_04(out_first_digit)
        out_first_digit = self.first_05(out_first_digit)
        out_first_digit = self.first_06(out_first_digit)
        out_first_digit = self.first_07(out_first_digit)
        out_first_digit = self.first_08(out_first_digit)
        out_first_digit = self.first_09(out_first_digit)
        out_first_digit = self.first_10(out_first_digit)
        out_first_digit = self.first_11(out_first_digit)
        out_first_digit = self.first_12(out_first_digit)
        out_first_digit = self.first_13(out_first_digit)
        out_first_digit = self.first_14(out_first_digit)
        out_first_digit = self.first_15(out_first_digit)
        out_first_digit = self.first_16(out_first_digit)
        out_first_digit = self.first_17(out_first_digit)
        out_first_digit = self.first_18(out_first_digit)
        out_first_digit = self.first_19(out_first_digit)
        out_first_digit = self.first_20(out_first_digit)
        out_first_digit = self.first_21(out_first_digit)
        out_first_digit = self.first_22(out_first_digit)
        out_first_digit = self.first_23(out_first_digit)
        out_first_digit = self.first_24(out_first_digit)

        out_second_digit = self.second_01(x)
        out_second_digit = self.second_02(out_second_digit)
        out_second_digit = self.second_03(out_second_digit)
        out_second_digit = self.second_04(out_second_digit)
        out_second_digit = self.second_05(out_second_digit)
        out_second_digit = self.second_06(out_second_digit)
        out_second_digit = self.second_07(out_second_digit)
        out_second_digit = self.second_08(out_second_digit)
        out_second_digit = self.second_09(out_second_digit)
        out_second_digit = self.second_10(out_second_digit)
        out_second_digit = self.second_11(out_second_digit)
        out_second_digit = self.second_12(out_second_digit)
        out_second_digit = self.second_13(out_second_digit)
        out_second_digit = self.second_14(out_second_digit)
        out_second_digit = self.second_15(out_second_digit)
        out_second_digit = self.second_16(out_second_digit)
        out_second_digit = self.second_17(out_second_digit)
        out_second_digit = self.second_18(out_second_digit)
        out_second_digit = self.second_19(out_second_digit)
        out_second_digit = self.second_20(out_second_digit)
        out_second_digit = self.second_21(out_second_digit)
        out_second_digit = self.second_22(out_second_digit)
        out_second_digit = self.second_23(out_second_digit)
        out_second_digit = self.second_24(out_second_digit)

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
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

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
