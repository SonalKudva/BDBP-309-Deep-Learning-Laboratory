import torch
from torch import nn

import math
import matplotlib.pyplot as plt


torch.manual_seed(111)

# PREPARING THE TRAINING DATA

# The training data is composed of pairs (x₁, x₂) so that x₂ consists of the value of the sine of x₁ for x₁ in the
# interval from 0 to 2π. You can implement it as follows:
train_data_length = 1024  # training set with 1024 pairs (x1, x2)
train_data = torch.zeros((train_data_length, 2)) # initialize train_data, a tensor with dimensions of 1024 rows and 2 columns
train_data[:, 0] =  2 * math.pi * torch.rand(train_data_length)  # use first column of train_data to store random values in interval of 0 to 2pi
train_data[:, 1] =  torch.sin(train_data[:, 0])  # calculate the sin of the first column
train_labels = torch.zeros(train_data_length)
train_set = [  # train_set as a list of tuples, with each row of train_data and train_labels represented in each tuple as expected in PyTorch's data loader
    (train_data[i], train_labels[i]) for i in range(train_data_length)]

plt.plot(train_data[:, 0], train_data[:, 1], ".")
plt.show()

# with train_set, we can create a PyTorch data loader
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)  # we have created train_loader which will shuffle the data from train_set and return batches of 32 samples that will be used to train the NN

# IMPLEMENTING THE DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self):  # in order to run this we need to call
        super().__init__()  # this
        self.model = nn.Sequential(
            nn.Linear(2, 256),  # 1st layer, I/P is 2D, first hidden layer is composed of 256 neurons with ReLu activation
            nn.ReLU(),
            nn.Dropout(0.3), # dropout layers are used to prevent overfitting
            nn.Linear(256, 128),  # 2nd layer, I/P is 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),  # 3rd layer, I/P is 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # 4th layer, I/P,is 64 with output composed of single neuron
            nn.Sigmoid(),
        )

    def forward(self, x):  # x represents the input of the model which is a 2D tensor
        output = self.model(x)
        return output

discriminator = Discriminator()

# IMPLEMENTING THE GENERATOR
# is the model that takes the samples from a latent space as its input and generates data resembling the data in the training set.
# In this case, it's a model with a two-dimensional input, which will receive random points (z1, z2) and a 2D output that must provide
# (x1bar, x2bar) points resembling those from the training data

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # output will consist of a vector with two elements having value ranging from -infinity to infinity
            # which will represent (x1bar, x2bar)
        )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()

# TRAINING THE MODELS
# Setting-up some parameters:
lr = 0.001  # learning rate which will be used to adapt the network weights
num_epochs = 300  # how many repetitions of training using the whole training set will be performed
loss_function = nn.BCELoss()  # assigns the variable loss_function

# Implementing the weight update rule for model training
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

# Implementing the training loop to feed the model the training samples and weights are updated to minimize the loss function
for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):  # get the real samples of the current batch from the data loader and assign them to real_samples
        # the first dimension of the tensor has number of elements equal to batch_size

        # data for training the discriminator
        real_sample_labels = torch.ones((batch_size, 1))  # .ones to create labels with value 1 for the real samples and then assign the labels to real_samples_labels
        latent_space_samples = torch.randn((batch_size, 2))  # You create the generated samples by storing random data in latent_space_samples
        generated_samples = generator(latent_space_samples)  # which you then feed to the generator to obtain generated_samples.
        generated_sample_labels = torch.zeros((batch_size, 1))  # ou use torch.zeros() to assign the value 0 to the labels for the generated samples, and then you store the labels in generated_samples_labels.
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_sample_labels, generated_sample_labels)
        )  # You concatenate the real and generated samples and labels and store them in all_samples and all_samples_labels, which you’ll use to train the discriminator.

        # training the discriminator
        discriminator.zero_grad()  # to clear all the gradients and avoid their accumulation
        output_discriminator = discriminator # calculate the output of the discriminator of all the samples
        loss_discriminator = loss_function(   # calculate the loss function using the output from the model in output_discriminator and the labels in all_samples_labels.
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()  # calculate the gradients to update the weights with loss_discriminator.backward()
        optimizer_discriminator.step()  #  update the discriminator weights by calling optimizer_discriminator.step().

        # data for training the generator
        latent_space_samples = torch.randn((batch_size, 2))

        # training the generator
        generator.zero_grad()  # clear the gradients with .zero_grad().
        generated_samples = generator(latent_space_samples)  # eed the generator with latent_space_samples and store its output in generated_samples.
        output_discriminator_generated = discriminator(generated_samples)  #  feed the generator’s output into the discriminator and store its output in output_discriminator_generated, which you’ll use as the output of the whole model.
        loss_generator = loss_function(
            output_discriminator_generated, real_sample_labels
        )  # # calculate the loss function using the output of the classification system stored in output_discriminator_generated and the labels in real_samples_labels, which are all equal to 1.
        loss_generator.backward()  # calculate the gradients and update the generator weights.
        optimizer_generator.step()
        # Remember that when you trained the generator, you kept the discriminator weights frozen since you created optimizer_generator with its first argument equal to generator.parameters()

        # show loss
        # Finally, on lines 35 to 37, you display the values of the discriminator and generator loss functions at the end of each ten epochs.
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

# CHECKING THE SAMPLES GENERATED BY THE GAN
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)

generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
plt.show()