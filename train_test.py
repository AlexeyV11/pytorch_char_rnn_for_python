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
    def __init__(self, rnn_type, hidden_size, num_layers=1, dropout_rate=0.0):
        super(MyRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # rnn layer
        self.rnn = rnn_type(input_size=1, hidden_size=hidden_size, num_layers=self.num_layers, dropout=dropout_rate)
        # linear layer for regression
        self.out = nn.Linear(hidden_size, 1)

    def init_hidden_state(self, batch_size):
        self.hidden_state = torch.zeros([self.num_layers, batch_size, self.hidden_size]).to(DEVICE)

    # input of shape (seq_len, batch, input_size):
    # hidden_state of shape (num_layers * num_directions, batch, hidden_size):
    def forward(self, x):
        result, self.hidden_state = self.rnn(x, self.hidden_state)
        result = self.out(result[:, :, :])
        return result

    def get_description(self):
        return self.rnn.__class__.__name__

class MyDataset(Dataset):

    # split is an array of [x0, ... , x_end]
    def __init__(self, path_to_data, seq_length, occurrence_min=1000):
        self._path_to_data = path_to_data
        self._seq_length = seq_length
        self._occurrence_min = 1000

        files = os.listdir(self._path_to_data)

        data_chunks = []
        for f in files:
            if f.endswith(".txt"):
                with open(os.path.join(self._path_to_data, f), 'r') as myfile:
                    code_tmp = myfile.read()

                    # data chunks for all the files
                    data_chunks.append(code_tmp)

        self._data = "".join(data_chunks)

    def compute_embeddings(self):
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

    def encode(self, str):
        return [self.char2int[s] for s in str]

    def decode(self, arr):
        return "".join([self.int2char[s] for s in arr])

    def __len__(self):
        return len(self._data) - self._seq_length - 1

    def __getitem__(self, idx):
            # [x0, .., xn]
            seq = self._data[idx:idx + self._seq_length + 1]


            encoded = self.encode(seq)

            return encoded[:-1], encoded[1:]


if False:
    def train_model(model, dataloader, loss_function, optimizer, batch_size, epochs, show_time=False, show_epoch_freq=10):
        model.train()
        loss_all = []

        time_total = 0
        for epoch in range(0,epochs):
            for x_batch, y_batch in dataloader:
                # new batch - new data so re-init hidden state + null the grads
                model.init_hidden_state(batch_size)
                optimizer.zero_grad()

                x_batch = x_batch[:,:,np.newaxis].permute([1, 0, 2])
                y_batch = y_batch[:,:,np.newaxis].permute([1, 0, 2])

                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

                start = time.time()

                # backpropogate results for all the timesteps
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimizer.step()

                time_total += time.time() - start

            loss_all.append(loss.cpu().data.numpy())

            if epoch % show_epoch_freq == 0:
                print("Training loss for epoch {} : ".format(epoch), loss.cpu().data.numpy())

        if show_time:
            print("Compute training time {}".format(time_total))


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


def experiment(rnn_type, seq_leng, batch_size, hidden_n, layers_n, learning_rate, epochs_n, show_time=False, show_graphs=False, show_epoch_freq=10):

    dataset = MyDataset("code_data", seq_leng)
    dataset.compute_embeddings()

    dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    #rnn = MyRNN(rnn_type, hidden_size=hidden_n, num_layers=layers_n).to(DEVICE)

    #optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    #loss_function = nn.MSELoss()

    #train_model(rnn, dataloader=dataloder, loss_function=loss_function, optimizer=optimizer, batch_size = batch_size, epochs=epochs_n, show_time=show_time, show_epoch_freq=show_epoch_freq)
    #test_model(rnn, dataloder, init_sequence_length=seq_leng, show_time=show_time, show_graphs=show_graphs)

def main():

    LEARNING_RATE = 0.005
    BATCH_SIZE = 100
    NUM_EPOCHS = 250
    SEQUENCE_LENGTH = 75
    HIDDEN_NEURONS=8
    NUM_LAYERS = 1

    experiment(nn.GRU, SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_NEURONS, NUM_LAYERS, LEARNING_RATE, NUM_EPOCHS, show_time=False, show_graphs=True, show_epoch_freq=25)


if __name__ == '__main__':
    main()