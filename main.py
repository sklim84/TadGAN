# coding: utf-8
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from _datasets.datasets import WADIDataset
from anomaly_detection import pw_reconstruction_error, detect_anomaly_with_threshold
from config import get_config

logging.basicConfig(
    filename=f'./logs/main_{datetime.now().strftime("%Y%m%d_%H:%M:%S")}.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def convert_to_windows(data, seq_len, stride=1):
    new_data = []
    for i in range(0, len(data) - seq_len, stride):
        _x = data[i:i + seq_len]
        new_data.append(_x)

    return np.array(new_data)


def critic_x_iteration(sample, device, seq_len, latent_space_dim, critic_x, decoder, optimizer):
    optimizer.zero_grad()

    x = sample['signal']
    x = x.to(device)

    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)

    valid_ones = torch.ones(valid_x.shape)
    valid_ones = valid_ones.to(device)
    critic_score_valid_x = torch.mean(valid_ones * valid_x)  # Wasserstein Loss

    # The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.empty(1, seq_len, latent_space_dim).uniform_(0, 1)
    z = z.to(device)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)

    fake_ones = torch.ones(fake_x.shape)
    fake_ones = fake_ones.to(device)
    critic_score_fake_x = torch.mean(fake_ones * fake_x)  # Wasserstein Loss

    alpha = torch.rand(x.shape)
    alpha = alpha.to(device)
    ix = Variable(alpha * x + (1 - alpha) * x_)  # Random Weighted Average
    ix.requires_grad_(True)
    v_ix = critic_x(ix)
    v_ix.mean().backward()
    gradients = ix.grad
    # Gradient Penalty Loss
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    # Critic has to maximize Cx(Valid X) - Cx(Fake X).
    # Maximizing the above is same as minimizing the negative.
    wl = critic_score_fake_x - critic_score_valid_x
    loss = wl + gp_loss
    loss.backward()
    optimizer.step()

    return loss


def critic_z_iteration(sample, device, seq_len, latent_space_dim, critic_z, encoder, optimizer):
    optimizer.zero_grad()

    x = sample['signal']
    x = x.to(device)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)

    valid_ones = torch.ones(valid_z.shape)
    valid_ones = valid_ones.to(device)
    critic_score_valid_z = torch.mean(valid_ones * valid_z)

    z_ = torch.empty(1, seq_len, latent_space_dim).uniform_(0, 1)
    z_ = z_.to(device)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    fake_ones = torch.ones(fake_z.shape)
    fake_ones = fake_ones.to(device)
    critic_score_fake_z = torch.mean(fake_ones * fake_z)  # Wasserstein Loss

    wl = critic_score_fake_z - critic_score_valid_z

    alpha = torch.rand(z.shape)
    alpha = alpha.to(device)
    iz = Variable(alpha * z + (1 - alpha) * z_)  # Random Weighted Average
    iz.requires_grad_(True)
    v_iz = critic_z(iz)
    v_iz.mean().backward()
    gradients = iz.grad
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    loss = wl + gp_loss
    loss.backward()
    optimizer.step()

    return loss


def encoder_iteration(sample, device, seq_len, latent_space_dim, critic_x, encoder, decoder, optimizer):
    optimizer.zero_grad()

    x = sample['signal']
    x = x.to(device)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    valid_ones = torch.ones(valid_x.shape)
    valid_ones = valid_ones.to(device)
    critic_score_valid_x = torch.mean(valid_ones * valid_x)  # Wasserstein Loss

    z = torch.empty(1, seq_len, latent_space_dim).uniform_(0, 1)
    z = z.to(device)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    fake_ones = torch.ones(fake_x.shape)
    fake_ones = fake_ones.to(device)
    critic_score_fake_x = torch.mean(fake_ones * fake_x)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    loss_function = torch.nn.MSELoss()
    mse = loss_function(x.float(), gen_x.float())
    loss_enc = mse + critic_score_valid_x - critic_score_fake_x
    loss_enc.backward(retain_graph=True)
    optimizer.step()

    return loss_enc


def decoder_iteration(sample, device, seq_len, latent_space_dim, critic_z, encoder, decoder, optimizer):
    optimizer.zero_grad()

    x = sample['signal']
    x = x.to(device)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    valid_ones = torch.ones(valid_z.shape)
    valid_ones = valid_ones.to(device)
    critic_score_valid_z = torch.mean(valid_ones * valid_z)

    z_ = torch.empty(1, seq_len, latent_space_dim).uniform_(0, 1)
    z_ = z_.to(device)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    fake_ones = torch.ones(fake_z.shape)
    fake_ones = fake_ones.to(device)
    critic_score_fake_z = torch.mean(fake_ones * fake_z)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    loss_function = torch.nn.MSELoss()
    mse = loss_function(x.float(), gen_x.float())
    loss_dec = mse + critic_score_valid_z - critic_score_fake_z
    loss_dec.backward(retain_graph=True)
    optimizer.step()

    return loss_dec


def train_model(encoder, decoder, critic_x, critic_z, optim_enc, optim_dec, optim_cx, optim_cz, data, batch_size,
                seq_len, n_critics, latent_space_dim, device, sampling_ratio=0.2):
    cx_nc_loss, cz_nc_loss = list(), list()

    # training data random sampling
    train_dataset = WADIDataset(data=data, sampling_ratio=sampling_ratio, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, drop_last=True)
    logging.info('Number of samples in train dataset {}'.format(len(train_dataset)))

    for i in range(1, n_critics + 1):
        cx_loss_list, cz_loss_list = list(), list()

        for batch, sample in enumerate(train_loader):
            cx_loss = critic_x_iteration(sample, device, seq_len, latent_space_dim, critic_x, decoder, optim_cx)
            cz_loss = critic_z_iteration(sample, device, seq_len, latent_space_dim, critic_z, encoder, optim_cz)
            cx_loss_list.append(cx_loss)
            cz_loss_list.append(cz_loss)

        cx_nc_loss.append(torch.mean(torch.tensor(cx_loss_list)))
        cz_nc_loss.append(torch.mean(torch.tensor(cz_loss_list)))

    logging.info('Critic training done')

    encoder_loss, decoder_loss = list(), list()

    for batch, sample in enumerate(train_loader):
        enc_loss = encoder_iteration(sample, device, seq_len, latent_space_dim, critic_x, encoder, decoder, optim_enc)
        dec_loss = decoder_iteration(sample, device, seq_len, latent_space_dim, critic_z, encoder, decoder, optim_dec)
        encoder_loss.append(enc_loss)
        decoder_loss.append(dec_loss)

    cx_loss_mean = torch.mean(torch.tensor(cx_nc_loss))
    cz_loss_mean = torch.mean(torch.tensor(cz_nc_loss))
    encoder_loss_mean = torch.mean(torch.tensor(encoder_loss))
    decoder_loss_mean = torch.mean(torch.tensor(decoder_loss))

    return cx_loss_mean, cz_loss_mean, encoder_loss_mean, decoder_loss_mean


def eval_model(encoder, decoder, critic_x, dataloader, batch_size, device):
    logging.info('Number of samples in test dataset {}'.format(len(dataloader.dataset)))

    reconstruction_error_list = list()
    critic_score_list = list()
    y_true = list()

    for batch, sample in enumerate(tqdm(dataloader)):
        signal = sample['signal'].to(device)
        anomaly = sample['anomaly']

        reconstructed_signal = decoder(encoder(signal))
        reconstructed_signal = torch.squeeze(reconstructed_signal)

        for i in range(0, batch_size):
            x_ = reconstructed_signal[i].detach().cpu().numpy()
            x = signal[i].cpu().numpy()
            y_true.append(int(anomaly[i].detach()))
            reconstruction_error = pw_reconstruction_error(x, x_)
            reconstruction_error_list.append(reconstruction_error)
        critic_score = torch.squeeze(critic_x(signal))
        critic_score_list.extend(critic_score.detach().cpu().numpy())

    # print(reconstruction_error_list)
    # print(critic_score_list)

    reconstruction_error_stats = stats.zscore(reconstruction_error_list)
    critic_score_stats = stats.zscore(critic_score_list)
    anomaly_score = reconstruction_error_stats * critic_score_stats

    # find best threshold
    threshold_list = list(np.arange(0, 1, 0.001))
    accuracy_list, precision_list, recall_list, f1score_list = [], [], [], []
    for threshold in threshold_list:
        y_predict = detect_anomaly_with_threshold(anomaly_score, threshold)
        accuracy_list.append(accuracy_score(y_true, y_predict))
        precision_list.append(precision_score(y_true, y_predict, pos_label=0))
        recall_list.append(recall_score(y_true, y_predict, pos_label=0))
        f1score_list.append(f1_score(y_true, y_predict, pos_label=0))

    df_result = pd.DataFrame(
        {'threshold': threshold_list, 'accuracy': accuracy_list, 'precision': precision_list, 'recall': recall_list,
         'f1score': f1score_list})
    df_result.sort_values(by=['f1score'], ascending=False, inplace=True)
    logging.info(df_result.iloc[0])

    return df_result


if __name__ == "__main__":

    # get arguments
    args = get_config()
    print(args)
    logging.info(args)

    # set path for datasets, model, and so on
    dir_datasets, dir_models, dir_results = './_datasets', './models', './results'

    # datasets
    if args.datasets == 'wadi':
        train_data_path = os.path.join(dir_datasets + '/WADI', 'train.npy')
        test_data_path = os.path.join(dir_datasets + '/WADI', 'test.npy')
        test_label_path = os.path.join(dir_datasets + '/WADI', 'labels.npy')
    elif args.datasets == 'swat':
        train_data_path = os.path.join(dir_datasets + '/SWaT', 'train.npy')
        test_data_path = os.path.join(dir_datasets + '/SWaT', 'test.npy')
        test_label_path = os.path.join(dir_datasets + '/SWaT', 'labels.npy')

    # model
    encoder_path = os.path.join(dir_models, f'encoder_{args.datasets}_{args.lr}_{args.latent_space_dim}.pt')
    encoder_opt_path = os.path.join(dir_models, f'encoder_opt_{args.datasets}_{args.lr}_{args.latent_space_dim}.pt')
    decoder_path = os.path.join(dir_models, f'decoder_{args.datasets}_{args.lr}_{args.latent_space_dim}.pt')
    decoder_opt_path = os.path.join(dir_models, f'decoder_opt_{args.datasets}_{args.lr}_{args.latent_space_dim}.pt')
    critic_x_path = os.path.join(dir_models, f'critic_x_{args.datasets}_{args.lr}_{args.latent_space_dim}.pt')
    critic_x_opt_path = os.path.join(dir_models, f'critic_x_opt_{args.datasets}_{args.lr}_{args.latent_space_dim}.pt')
    critic_z_path = os.path.join(dir_models, f'critic_z_{args.datasets}_{args.lr}_{args.latent_space_dim}.pt')
    critic_z_opt_path = os.path.join(dir_models, f'critic_z_opt_{args.datasets}_{args.lr}_{args.latent_space_dim}.pt')

    # test result
    test_result_path = os.path.join(dir_results, f'result_{args.datasets}_{args.lr}_{args.latent_space_dim}.csv')

    # fix randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)

    # GPU setting
    gpu = 'cuda:' + args.device
    device = torch.device(gpu)

    # Train
    if args.mode == 'all' or args.mode == 'train':
        logging.info('Start training')

        # load data
        train_data = np.load(train_data_path)
        signal_shape = train_data.shape[1]

        # create model
        encoder = model.Encoder(signal_shape, args.latent_space_dim).to(device)
        decoder = model.Decoder(signal_shape, args.latent_space_dim).to(device)
        critic_x = model.CriticX(signal_shape).to(device)
        critic_z = model.CriticZ(args.latent_space_dim).to(device)

        optim_enc = optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        optim_dec = optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        optim_cx = optim.Adam(critic_x.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        optim_cz = optim.Adam(critic_z.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        # train
        cx_epoch_loss, cz_epoch_loss, encoder_epoch_loss, decoder_epoch_loss = list(), list(), list(), list()
        for epoch in tqdm(range(args.epoch)):
            logging.info('Epoch {}'.format(epoch))

            cx_loss_mean, cz_loss_mean, encoder_loss_mean, decoder_loss_mean \
                = train_model(encoder, decoder, critic_x, critic_z, optim_enc, optim_dec, optim_cx, optim_cz,
                              train_data, args.batch, args.seq_len, args.n_critics, args.latent_space_dim, device, 0.2)

            cx_epoch_loss.append(cx_loss_mean)
            cz_epoch_loss.append(cz_loss_mean)
            encoder_epoch_loss.append(encoder_loss_mean)
            decoder_epoch_loss.append(decoder_loss_mean)
            logging.info('Encoder decoder training done in epoch {}'.format(epoch))
            logging.info('critic x loss {:.3f} critic z loss {:.3f} encoder loss {:.3f} decoder loss {:.3f}\n'.format(
                cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

            # save model
            if epoch % 10 == 0:
                torch.save(encoder.state_dict(), encoder_path)
                torch.save(decoder.state_dict(), decoder_path)
                torch.save(critic_x.state_dict(), critic_x_path)
                torch.save(critic_z.state_dict(), critic_z_path)
                torch.save(optim_enc.state_dict(), encoder_opt_path)
                torch.save(optim_dec.state_dict(), decoder_opt_path)
                torch.save(optim_cx.state_dict(), critic_x_opt_path)
                torch.save(optim_cz.state_dict(), critic_z_opt_path)

    # Train
    elif args.mode == 'all' or args.mode == 'test':
        logging.info('Start testing')

        # load data
        test_data = np.load(test_data_path)
        test_label = np.load(test_label_path)
        signal_shape = test_data.shape[1]

        test_dataset = WADIDataset(data=test_data, label=test_label)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=8, drop_last=True)

        # load model
        device = 'cpu'
        encoder = model.Encoder(signal_shape, args.latent_space_dim).to(device)
        encoder_state = torch.load(encoder_path)
        # print(encoder_state)
        encoder.load_state_dict(encoder_state)

        decoder = model.Decoder(signal_shape, args.latent_space_dim).to(device)
        decoder.load_state_dict(torch.load(decoder_path))
        critic_x = model.CriticX(signal_shape).to(device)
        critic_x.load_state_dict(torch.load(critic_x_path))

        df_result = eval_model(encoder, decoder, critic_x, test_loader, args.batch, device)
        df_result.to_csv(test_result_path)
