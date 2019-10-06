import os
import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import matplotlib.pyplot as plt
# from skimage import io, transform, util
# from scipy.fftpack import fft2
# from sklearn.preprocessing import RobustScaler, MinMaxScaler
from pp_dataset import PlannerPortfolioDataset
from architectures import PlaNet
import argparse
import logging
import time
torch.manual_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

""" GET PATH INFORMATION """
CURRENT_DIR = os.getcwd()
processed_df_path = os.path.join(CURRENT_DIR, 'df.csv')
image_folder_path = os.path.join(CURRENT_DIR, 'IPC-image-data/lifted/')

""" OPTIONS """
key_list = ['net_1', 'net_1_double', 'net_1_triple', 'net_2', 'net_2_double',
            'net_2_triple', 'net_3', 'net_3_double', 'net_3_triple']


""" SETUP PARSER """
parser = argparse.ArgumentParser()
parser.add_argument('net_key', type=str, help='the key of the desired net.',
                    choices=key_list)
parser.add_argument('use_ft', type=int, help='use fourier transform on the data.', choices=list(range(2)))
parser.add_argument('-optimizer', type=str, help='what optim to use?', choices=['Adam', 'SGD'], default='Adam')
parser.add_argument('-epochs', type=int, help='ho many epochs to perform', default=1000)
parser.add_argument('-batch', type=int, help='ho many batches', default=4)
parser.add_argument('-lr', type=float, help='learning rate', default=0.00001)
parser.add_argument('-betas', type=float, help='betas for Adam optim', default=(0.9,0.9999))
parser.add_argument('-momentum', type=float, help='momentum for SGD optim', default=0.9)

if __name__ == '__main__':
    args = parser.parse_args()  # Disable during debugging

    # args = argparse.Namespace(net_key='net_3_triple', use_ft=False, optimizer='Adam', epochs=100, batch=16,
    #                           lr=0.000001, betas=(0.9,0.999)) # For debugging purposes

    """ SETUP EXPERIMENT """
    path_to_model = 'saved_models/' + args.net_key + '_' +  args.optimizer + '_'
    if args.use_ft:
        path_to_model = path_to_model + 'fft.pt'
    else:
        path_to_model = path_to_model + 'no-fft.pt'

    log_file = path_to_model.replace('.pt', '.log')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(log_file, '+w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


    if args.optimizer == 'Adam':
        optimizer = (optim.Adam, {'lr':args.lr, 'betas':args.betas})
    elif args.optimizer == 'SGD':
        optimizer = (optim.SGD, {'lr':args.lr, 'momentum':args.momentum})


    exp_dict = {
        'net_key': args.net_key,
        'fourier': args.use_ft,
        'optimizer': optimizer,
        'max_num_epochs': args.epochs,
        'path_to_model': os.path.join(CURRENT_DIR, path_to_model)
    }

    logger.info(f'Starting CNN training...')
    logger.info(f'Options: \n{exp_dict}')

    """ ESTABLISH DATASET """
    plan_dataset = PlannerPortfolioDataset(processed_df_path, image_folder_path, ftransform=exp_dict['fourier'])

    # # test dataset
    # sample = plan_dataset[0]

    """ ESTABLISH A DATA LOADER """
    dataloader = DataLoader(plan_dataset, batch_size=4, shuffle=True)

    # # test dataloader
    # image, planner_results = next(iter(dataloader))

    """ ESTABLISH A NETWORK """
    net_key = exp_dict['net_key']
    input_size = tuple(next(iter(dataloader))[0].shape)
    net = PlaNet(net_key, input_size=input_size)
    net = net.to(device)
    logger.info(f'CNN established:\n{net}')


    # # test it with a random input
    # # input = torch.randn(1, 1, 128, 128) # 1 image of 1 chanel and 128x128 height and width
    # input, _ = next(iter(dataloader))
    # out = net(input)
    # print(out.shape)

    """ ESTABLISH A LOSS FUNCTION """
    criterion = nn.BCELoss()
    criterion.to(device)
    logger.info(f'Loss function established: {criterion}')

    # # test loss function
    # input, target = next(iter(dataloader)).values()
    # output = net(input)
    # loss = criterion(output, target)
    # print(loss)

    """ ESTABLISH AN OPTIMIZER """
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer, cfg = exp_dict['optimizer']
    optimizer = optimizer(net.parameters(), **cfg)
    logger.info(f'Optimizer established: {criterion}')

    # # test optimizer:
    # input, target = next(iter(dataloader)).values()
    # optimizer.zero_grad()   # zero the gradient buffers
    # output = net(input)
    # loss = criterion(output, target)
    # print(loss)
    # # update the weights
    # loss.backward()
    # optimizer.step()    # Does the update
    # # recalc the output and loss
    # output = net(input)
    # loss = criterion(output, target)
    # print(loss)

    """ TRAIN """
    train_size = int(0.6 * len(plan_dataset))
    valid_size = int(0.2 * len(plan_dataset))
    test_size = len(plan_dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(plan_dataset,
                                                                               [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    best_acc = 0
    logger.info(f'Training Initiated.')

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    best_valid_loss = 10.0
    patience = 15
    epochs_without_improvement = 0

    for epoch in range(exp_dict['max_num_epochs']):  # loop over the dataset multiple times
        t0 = time.time()
        ###################
        # train the model #
        ###################
        net.train()
        running_loss = 0.0
        for i, sample in enumerate(train_loader):
            inputs, targets = sample
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        net.eval() # prep model for evaluation
        for i, sample in enumerate(valid_loader):
            inputs, targets = sample
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = net(inputs)
            # calculate the loss
            loss = criterion(outputs, targets)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_time = round(time.time() - t0, 2)

        print_msg = (f'[{epoch}/{exp_dict["max_num_epochs"]}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'epoch time: {epoch_time}')

        logger.info(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss*0.99
            logger.info('New best model, saving ...')
            torch.save(net, exp_dict['path_to_model'])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > patience:
            logger.info("Early stopping")
            logger.info(f"train loss: {avg_train_losses}")
            logger.info(f"valid loss: {avg_valid_losses}")
            break

    """ TEST BEST MODEL """
    logger.info(f'\nBest Model Test Initiated.')
    net = torch.load(exp_dict['path_to_model'])
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            for i, idx in enumerate(torch.argmax(outputs, dim=1)):
                total +=1
                correct += int(labels[i,idx])
        acc = correct/total
    logger.info(f'\nBest Model Accuracy: %{round(acc*100)}.')
    print(' ')
