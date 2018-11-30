import random
import itertools
import collections
import time
import os

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# TODO: FIXME: check if we can improve this and set it globally instead of transfering some particular variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyRNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout_rate=0.05):
        super(MyRNN, self).__init__()

        # we input and output one hot character
        self._input_size = input_size
        self._output_size = input_size

        self._hidden_size = hidden_size
        self._num_layers = num_layers

        # rnn layer
        self.rnn = rnn_type(input_size=self._input_size, hidden_size=self._hidden_size, num_layers=self._num_layers, dropout=dropout_rate)
        # linear layer for regression
        self.out = nn.Linear(self._hidden_size, self._output_size )

        self._hidden_state = None

    def init_hidden_state(self, batch_size):
        self._hidden_state = torch.zeros([self._num_layers, batch_size, self._hidden_size]).to(DEVICE)

    # input of shape (seq_len, batch, input_size):
    # hidden_state of shape (num_layers * num_directions, batch, hidden_size):
    def forward(self, x):
        result, self._hidden_state = self.rnn(x, self._hidden_state)
        result = self.out(result[:, :, :])
        return result

    def get_description(self):
        return self.rnn.__class__.__name__

class MyDataset(Dataset):

    # split is an array of [x0, ... , x_end]
    def __init__(self, path_to_data, seq_length, occurrence_min=1000, one_hot_mode=True):
        self._path_to_data = path_to_data
        self._seq_length = seq_length
        self._occurrence_min = 1000
        self._one_hot_mode = one_hot_mode

        files = os.listdir(self._path_to_data)

        data_chunks = []
        for f in files:
            if f.endswith(".txt"):
                with open(os.path.join(self._path_to_data, f), 'r') as myfile:
                    code_tmp = myfile.read()

                    # data chunks for all the files
                    data_chunks.append(code_tmp)

        self._data = "".join(data_chunks)
        self._init_embeddings()

    def _init_embeddings(self):
        print("size of code database is {} characters".format(len(self._data)))

        cntr = collections.Counter(self._data)
        freq_chars = [(char, count) for char, count in cntr.most_common() if count > self._occurrence_min]

        print("list of characters with occurance more than {}".format(self._occurrence_min))
        for char, cnt in freq_chars:
            if char == '\n':
                char = "newline"
            elif char == ' ':
                char = 'space'
            elif char == '\t':
                char = 'tab'

            print("{:20}{:}".format(char, cnt))



        self.int2char = dict(enumerate([c for c, _ in freq_chars]))

        # len(self.int2char) - will be code for unknown
        unknown_code = len(self.int2char)
        unknown_char = '?'

        # by default we will return unknown code
        self.char2int = collections.defaultdict(lambda: unknown_code, {ch: ii for ii, ch in self.int2char.items()})

        # add unknown to dict in case if RNN generate one
        self.int2char[unknown_code] = unknown_char

    def get_class_count(self):
        # or len(self.char2int) + 1
        return len(self.int2char)

    def _str_to_nums(self, str):
        return [self.char2int[s] for s in str]

    def encode(self, str):
        encoded = np.array(self._str_to_nums(str), dtype=np.float32)

        if self._one_hot_mode:
            encoded = self.one_hot_encode(encoded)

        return encoded

    def decode(self, arr):
        if self._one_hot_mode:
            arr = np.argmax(arr, axis=-1)

        arr = arr.flatten().tolist()
        return "".join([self.int2char[s] for s in arr])

    def __len__(self):
        return len(self._data) - self._seq_length - 1

    def __getitem__(self, idx):
        # 1 extra leng to generate input + output
        encoded = self.encode(self._data[idx:idx + self._seq_length + 1])

        return encoded[:-1,:], encoded[1:,:]

    def one_hot_encode(self, batch):

        # Initialize the the encoded array
        one_hot = np.zeros((batch.size, self.get_class_count()), dtype=np.float32)

        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), batch.flatten().astype(dtype=np.int)] = 1.

        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*batch.shape, self.get_class_count()))

        return one_hot


def train_model(model, dataloader, loss_function, optimizer, batch_size, epochs, show_time=False, show_iterations=10):
    model.train()
    loss_all = []

    time_total = 0
    iter = 0
    for epoch in range(0,epochs):
        for x_batch, y_batch in dataloader:
            # new batch - new data so re-init hidden state + null the grads
            model.init_hidden_state(batch_size)
            optimizer.zero_grad()

            x_batch = x_batch.permute([1, 0, 2])
            y_batch = y_batch.permute([1, 0, 2])

            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            start = time.time()

            # backpropogate results for all the timesteps
            output = model(x_batch)


            #
            # TODO: should we process just last half?
            #
            loss_total = 0
            for c in range(y_batch.shape[0]):
                #loss_total += loss_function(output[c, :, :], y_batch[c, :, :].type(torch.LongTensor).to(DEVICE))
                loss_total += loss_function(output[c, :, :], torch.max(y_batch[c, :, :], 1)[1].to(DEVICE))



            loss_total.backward()
            optimizer.step()

            time_total += time.time() - start

            loss_all.append(loss_total.cpu().data.numpy())

            if iter % show_iterations == 0:
                print("Training loss for iter {} : ".format(iter), loss_total.data.cpu().numpy())

                generate_string(model, dataloader.dataset)
            iter += 1

    if show_time:
        print("Compute training time {}".format(time_total))


def choose_next_max(arr):
    probs = nn.Softmax()(arr.squeeze())

    zer = np.zeros(arr.shape)
    index = int(torch.argmax(probs).cpu().numpy())

    zer[0, 0, index] = 1.0

    return zer


def generate_string(model, dataset):
    model.eval()

    str = "<BOF>\n"
    str_enc = dataset.encode(str)

    next = torch.Tensor(str_enc[np.newaxis, :, :]).cuda().permute([1, 0, 2])

    model.init_hidden_state(1)

    for i in range(200):
        out = model.forward(next)

        # take last result; add dimension
        next = choose_next_max(out[-1,:,:][np.newaxis,:,:])

        str += dataset.decode(next)

        next = torch.Tensor(next).cuda()

    print(str)
    print()
    model.train()
if False:

    def test_model(model, dataloader, init_sequence_length, show_time=False, show_graphs=False):
        model.eval()

        model.init_hidden_state(1)

        batch_input, batch_y = dataloader.dataset[0]

        initial_input = torch.Tensor(batch_input[:, np.newaxis, np.newaxis]).to(DEVICE)

        final_outputs = []

        # start with initial sequence
        output = model(initial_input)
        output = output[-1, :, :]
        final_outputs.append(output.cpu().data.squeeze_())
        output = output[np.newaxis, :, :]

        time_total = 0
        for _ in range(len(dataloader.dataset.split)-init_sequence_length):
            start = time.time()
            # here we are using our current result as an input to get next one
            output = model(output)
            time_total += time.time() - start

            final_outputs.append(output.cpu().data.squeeze_())

        if show_time:
            print("Compute testing time {}".format(time_total))

        if show_graphs:
            def myplot(points, label_name):
                plt.plot(points, linestyle='--', marker='.', label=label_name)

            myplot(dataloader.dataset.split[init_sequence_length:], 'actual')
            myplot(final_outputs, 'predicted')

            plt.legend(bbox_to_anchor=(.90, 1.05), loc=2, borderaxespad=0.)
            plt.savefig("sin_wave_{}.png".format(model.get_description()))
            plt.close()
            #plt.show()

        return final_outputs


def experiment(rnn_type, seq_leng, batch_size, hidden_n, layers_n, learning_rate, epochs_n, show_time=False, show_graphs=False, show_iterations=100):

    dataset = MyDataset("code_data", seq_leng)

    dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    rnn = MyRNN(rnn_type, input_size=dataset.get_class_count(), hidden_size=hidden_n, num_layers=layers_n).to(DEVICE)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_model(rnn, dataloader=dataloder, loss_function=loss_function, optimizer=optimizer, batch_size = batch_size, epochs=epochs_n, show_time=show_time, show_iterations=show_iterations)

    #test_model(rnn, dataloder, init_sequence_length=seq_leng, show_time=show_time, show_graphs=show_graphs)

def main():

    LEARNING_RATE = 0.005
    BATCH_SIZE = 100
    NUM_EPOCHS = 250
    SEQUENCE_LENGTH = 200
    HIDDEN_NEURONS=100
    NUM_LAYERS = 2

    experiment(nn.GRU, SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_NEURONS, NUM_LAYERS, LEARNING_RATE, NUM_EPOCHS, show_time=False, show_graphs=True, show_iterations=500)


if __name__ == '__main__':
    main()