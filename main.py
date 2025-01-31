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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import model
from _datasets.datasets import TadGANDataset

from anomaly_detection import pw_reconstruction_error, detect_anomaly_with_threshold, detect_anomaly, \
    prune_false_positive, dtw_reconstruction_error
from config import get_config
import nvidia_smi
from utils import create_dataloaders

logging.basicConfig(
    filename=f'./logs/main_{datetime.now().strftime("%Y%m%d")}.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()


def critic_x_iteration(x, device, batch_size, latent_space_dim, critic_x, decoder, optimizer):
    optimizer.zero_grad()

    x = x.view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)

    valid_ones = torch.ones(valid_x.shape)
    valid_ones = valid_ones.to(device)
    critic_score_valid_x = torch.mean(valid_ones * valid_x)  # Wasserstein Loss

    # The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
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


def critic_z_iteration(x, device, batch_size, latent_space_dim, critic_z, encoder, optimizer):
    optimizer.zero_grad()

    x = x.view(1, batch_size, signal_shape)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)

    valid_ones = torch.ones(valid_z.shape)
    valid_ones = valid_ones.to(device)
    critic_score_valid_z = torch.mean(valid_ones * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
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


def encoder_iteration(x, device, batch_size, latent_space_dim, critic_x, encoder, decoder, optimizer):
    optimizer.zero_grad()

    x = x.view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    valid_ones = torch.ones(valid_x.shape)
    valid_ones = valid_ones.to(device)
    critic_score_valid_x = torch.mean(valid_ones * valid_x)  # Wasserstein Loss

    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
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
    # loss_enc.backward(retain_graph=True)
    loss_enc.backward()
    optimizer.step()

    return loss_enc


def decoder_iteration(x, device, batch_size, latent_space_dim, critic_z, encoder, decoder, optimizer):
    optimizer.zero_grad()

    x = x.view(1, batch_size, signal_shape)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    valid_ones = torch.ones(valid_z.shape)
    valid_ones = valid_ones.to(device)
    critic_score_valid_z = torch.mean(valid_ones * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
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
    # loss_dec.backward(retain_graph=True)
    loss_dec.backward()
    optimizer.step()

    return loss_dec


def train_model(encoder, decoder, critic_x, critic_z, optim_enc, optim_dec, optim_cx, optim_cz, train_loaders,
                batch_size, n_critics, latent_space_dim, device):
    cx_nc_loss, cz_nc_loss = list(), list()

    for t, train_loader in enumerate(train_loaders):
        logging.info(f'Number of batches in #{t} train loader: {len(train_loader)}')
        logging.info('Critic training start')

        for i in range(1, n_critics + 1):
            cx_loss_list, cz_loss_list = list(), list()

            for batch, sample in enumerate(train_loader):
                x = sample['signal']
                x = torch.Tensor(x).to(device)

                cx_loss = critic_x_iteration(x, device, batch_size, latent_space_dim, critic_x, decoder, optim_cx)
                cz_loss = critic_z_iteration(x, device, batch_size, latent_space_dim, critic_z, encoder, optim_cz)
                cx_loss_list.append(cx_loss)
                cz_loss_list.append(cz_loss)

            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss_list)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss_list)))
            logging.info('#{} Critic training done'.format(i))

        logging.info('Encoder decoder training start')
        encoder_loss, decoder_loss = list(), list()
        for batch, sample in enumerate(train_loader):
            x = sample['signal']
            x = torch.Tensor(x).to(device)

            if batch % 1000 == 0:
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(int(args.device))
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                logging.info("batch: {}, Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)"
                             .format(batch, args.device, nvidia_smi.nvmlDeviceGetName(handle),
                                     100 * info.free / info.total,
                                     info.total, info.free, info.used))

            enc_loss = encoder_iteration(x, device, batch_size, latent_space_dim, critic_x, encoder, decoder, optim_enc)
            dec_loss = decoder_iteration(x, device, batch_size, latent_space_dim, critic_z, encoder, decoder, optim_dec)
            encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)

        logging.info('Encoder decoder training done')

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

            # reconstruction_error = pw_reconstruction_error(x, x_)
            reconstruction_error = dtw_reconstruction_error(x, x_)
            reconstruction_error_list.append(reconstruction_error)
        critic_score = torch.squeeze(critic_x(signal))
        critic_score_list.extend(critic_score.detach().cpu().numpy())

    reconstruction_error_stats = stats.zscore(reconstruction_error_list)
    critic_score_stats = stats.zscore(critic_score_list)
    anomaly_score = reconstruction_error_stats * critic_score_stats

    y_predict = detect_anomaly(anomaly_score)
    y_predict = prune_false_positive(y_predict, anomaly_score, change_threshold=0.1)

    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict, pos_label=0)
    recall = recall_score(y_true, y_predict, pos_label=0)
    f1score = f1_score(y_true, y_predict, pos_label=0)

    logging.info(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1score: {f1score}')

    return accuracy, precision, recall, f1score


if __name__ == "__main__":

    # get arguments
    args = get_config()
    logging.info(args)

    # set path for datasets, model, and so on
    dir_models, dir_results = './models', './results'

    # datasets
    train_loaders, test_loader, signal_shape = create_dataloaders(datasets=args.datasets, batch_size=args.batch,
                                                                  sampling_ratio=args.sampling_ratio)

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
    logging.info(device)

    # Train
    if args.mode == 'all' or args.mode == 'train':
        logging.info('Start training {}'.format(args.datasets))

        # create model
        encoder = model.Encoder(signal_shape, args.latent_space_dim)
        # encoder = torch.nn.DataParallel(encoder)
        encoder = encoder.to(device)
        decoder = model.Decoder(signal_shape, args.latent_space_dim)
        # decoder = torch.nn.DataParallel(decoder)
        decoder = decoder.to(device)
        critic_x = model.CriticX(signal_shape)
        # critic_x = torch.nn.DataParallel(critic_x)
        critic_x = critic_x.to(device)
        critic_z = model.CriticZ(args.latent_space_dim)
        # critic_z = torch.nn.DataParallel(critic_z)
        critic_Z = critic_z.to(device)

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
                              train_loaders, args.batch, args.n_critics, args.latent_space_dim, device)

            cx_epoch_loss.append(cx_loss_mean)
            cz_epoch_loss.append(cz_loss_mean)
            encoder_epoch_loss.append(encoder_loss_mean)
            decoder_epoch_loss.append(decoder_loss_mean)
            logging.info('Training done in epoch {}'.format(epoch))
            logging.info('critic x loss {:.3f} critic z loss {:.3f} encoder loss {:.3f} decoder loss {:.3f}\n'.format(
                cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

            # save model
            if (epoch + 1) % 10 == 0:
                # Saving torch.nn.DataParallel Models
                # torch.save(encoder.module.state_dict(), encoder_path)
                # torch.save(decoder.module.state_dict(), decoder_path)
                # torch.save(critic_x.module.state_dict(), critic_x_path)
                # torch.save(critic_z.module.state_dict(), critic_z_path)
                torch.save(encoder.state_dict(), encoder_path)
                torch.save(decoder.state_dict(), decoder_path)
                torch.save(critic_x.state_dict(), critic_x_path)
                torch.save(critic_z.state_dict(), critic_z_path)
                torch.save(optim_enc.state_dict(), encoder_opt_path)
                torch.save(optim_dec.state_dict(), decoder_opt_path)
                torch.save(optim_cx.state_dict(), critic_x_opt_path)
                torch.save(optim_cz.state_dict(), critic_z_opt_path)

        logging.info('End training {}'.format(args.datasets))

    # Train
    if args.mode == 'all' or args.mode == 'test':
        logging.info('Start testing {}'.format(args.datasets))

        # load model
        encoder = model.Encoder(signal_shape, args.latent_space_dim).to(device)
        encoder.load_state_dict(torch.load(encoder_path))
        decoder = model.Decoder(signal_shape, args.latent_space_dim).to(device)
        decoder.load_state_dict(torch.load(decoder_path))
        critic_x = model.CriticX(signal_shape).to(device)
        critic_x.load_state_dict(torch.load(critic_x_path))

        accuracy, precision, recall, f1score = eval_model(encoder, decoder, critic_x, test_loader, args.batch, device)

        df_result = pd.DataFrame(
            {'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1score': [f1score]})
        df_result.to_csv(test_result_path)
