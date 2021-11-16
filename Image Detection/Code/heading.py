import os
import numpy as np
import torch as t
from torch import nn
#from torchviz import make_dot
#import graphviz
import math
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt

def resize_image(input_size = 28):
    transform = transforms.Compose([
    transforms.Resize(input_size), 
    transforms.ToTensor()
    ])
    return transform

def load_EMNIST_data(transform, BATCH_SIZE, train, PATH = 'Data', split = 'mnist'):
    load_dataset = datasets.EMNIST(PATH, 
                               split = 'mnist',
                               train=train,
                               download=True,
                               transform=transform)

    data_loader = t.utils.data.DataLoader(dataset = load_dataset, 
                                            batch_size = BATCH_SIZE, shuffle = True)


    return data_loader

def view_datasets(image_loader):

    images, labels = next(iter(image_loader))
    img = torchvision.utils.make_grid(images)

    img = img.numpy().transpose(1, 2, 0)
    print(labels.numpy())
    plt.axis('off')
    plt.imshow(img)
    return (images, labels)

def create_network(network):
    device = t.device('cpu')

    res = network
    net = res.to(device)

    return net

def train_model(net, train_loader, LR, epochs = 1, number_of_images = None):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(),
        lr=LR
    )
    for epoch in range(epochs):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cpu(), Variable(labels).cpu()
            optimizer.zero_grad()  #Make gradient to zero
            outputs = net(inputs)  #Forward calculation 
            loss = loss_function(outputs, labels)  #Get loss function
            loss.backward()  #back propogation
            optimizer.step()  #Update parameter.
            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
            if number_of_images is None:
                pass
            else:
                if i * train_loader.batch_size >= number_of_images:
                    print ("Current Loss {}".format(sum_loss))
                    break
    return net

def test_model(net, test_loader, number_of_images = None):
    net.eval()  #Convert to test model
    correct = 0
    total = 0
    for i, data_test in enumerate(test_loader):
        images, labels = data_test
        images, labels = Variable(images).cpu(), Variable(labels).cpu()
        output_test = net(images)
        _, predicted = t.max(output_test, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if number_of_images is None:
            pass
        else:
            if i * test_loader.batch_size >= number_of_images:
                print("correct1: ", correct)
                print("Test acc: {0}".format(correct.item() /
                                     i * test_loader.batch_size))
                break
    if number_of_images is None:
        print("correct1: ", correct)
        print("Test acc: {0}".format(correct.item() /
                                     len(test_loader.dataset)))

def predict_image(net, image_size = 28, num_of_prediction = 1, input_image = None):

    if input_image is None:
        transform = resize_image(image_size)
        predict_loader = load_EMNIST_data(transform, num_of_prediction, train = False, PATH = 'Data')
        images, labels = view_datasets(predict_loader)

        images = Variable(images).cpu()
    else:
        input_image = input_image.unsqueeze(-3)
        img = torchvision.utils.make_grid(input_image)
        img = img.numpy().transpose(1, 2, 0)
        plt.imshow(img)
        images = Variable(input_image).cpu()
    output_test = net(images)
    _, predicted = t.max(output_test, 1)

    print ("The predict result is {}".format(predicted.numpy()))

def train_RNN(net_work, hidden_size, num_of_iter = 1000, num_time_steps = 50):
    net_work = net_work
    hidden_size = hidden_size
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net_work.parameters(), 1e-2)

    hidden_prev = t.zeros(1, 1, hidden_size)

    for iter in range(num_of_iter):

        start = np.random.randint(3, size=1)[0]
        time_steps = np.linspace(start, start + 10, num_time_steps)
        data = np.sin(time_steps)
        data = data.reshape(num_time_steps, 1)
        x = t.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
        y = t.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

        output, hidden_prev = net_work(x, hidden_prev)
        hidden_prev = hidden_prev.detach()
        loss = criterion(output, y)
        net_work.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("Iter: {} loss: {} ".format(iter, loss))

def test_RNN(net_work, hidden_size, num_time_steps = 50):
    model = net_work
    hidden_size = hidden_size
    criterion = nn.MSELoss()
    hidden_prev = t.zeros(1, 1, hidden_size)
    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = t.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    y = t.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)
    output, hidden = model(x, hidden_prev)
    loss = criterion(output, y)
    print("Loss: {} ".format(loss))

    prediction = []
    input = x[:, 0, :]
    for _ in range(x.shape[1]):
        input = input.view(1, 1, 1)
        (pred, hidden_prev) = model(input, hidden_prev)
        input = pred
        prediction.append(pred.detach().numpy().ravel()[0])
    x = x.data.numpy().ravel()
    y = y.data.numpy()
    plt.scatter(time_steps[:-1], x.ravel(), s=90)
    plt.plot(time_steps[:-1], x.ravel())

    plt.scatter(time_steps[1:], prediction)
    plt.show()