# -*- coding: utf-8 -*-
# @Time    : 2020/12/25
# @Author  : Shuyu ZHANG
# @FileName: Revised.py
# @Software: PyCharm
# @Description: Revised Code with NNI


import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

#===================================================
import nni
from nni.utils import merge_parameter
#---------------------------------------------------

logger = logging.getLogger('mnist_Revised')
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if (args['batch_num'] is not None) and batch_idx >= args['batch_num']:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def main(args):
    if args['device'] != -1 and torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(args['device']))
    else:
        device = torch.device('cpu')

    data_dir = args['data_dir']

    train_loader = data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args['batch_size'], shuffle=True)

    test_loader = data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    hidden_size = args['hidden_size']

    model = Net(hidden_size=hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)

        logger.debug('test accuracy %g', test_acc)
        logger.debug('Pipe send intermediate result done.')

#===================================================
        # report intermediate result
        nni.report_intermediate_result(test_acc)
#---------------------------------------------------

    logger.debug('Final result is %g', test_acc)
    logger.debug('Send final result done.')

#===================================================
    # report final result
    nni.report_final_result(test_acc)
#---------------------------------------------------


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--device', type=int, default=0,
                        help='selected device for training -1:cpu')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
#===================================================
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
#---------------------------------------------------