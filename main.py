# coding: utf-8
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy import stats
from anomaly_detection import find_scores, dtw_reconstruction_error, detect_anomaly, prune_false_positive
import model
from _datasets.datasets import WADIDataset
from config import get_config
import random
from tqdm import tqdm

logging.basicConfig(
    filename='train.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def critic_x_iteration(sample, device, batch_size, latent_space_dim, critic_x, decoder, optimizer):
    optimizer.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape)
    x = x.to(device)

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


def critic_z_iteration(sample, device, batch_size, latent_space_dim, critic_z, encoder, optimizer):
    optimizer.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape)
    x = x.to(device)
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


def encoder_iteration(sample, device, batch_size, latent_space_dim, critic_x, encoder, decoder, optimizer):
    optimizer.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape)
    x = x.to(device)
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
    loss_enc.backward(retain_graph=True)
    optimizer.step()

    return loss_enc


def decoder_iteration(sample, device, batch_size, latent_space_dim, critic_z, encoder, decoder, optimizer):
    optimizer.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape)
    x = x.to(device)
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
    loss_dec.backward(retain_graph=True)
    optimizer.step()

    return loss_dec


if __name__ == "__main__":

    args = get_config()
    print(args)

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

    signal_shape = 123  # WADI
    encoder_path = 'models/encoder.pt'
    encoder_opt_path = 'models/encoder_opt.pt'
    decoder_path = 'models/decoder.pt'
    decoder_opt_path = 'models/decoder_opt.pt'
    critic_x_path = 'models/critic_x.pt'
    critic_x_opt_path = 'models/critic_x_opt.pt'
    critic_z_path = 'models/critic_z.pt'
    critic_z_opt_path = 'models/critic_z_opt.pt'

    encoder = model.Encoder(encoder_path, signal_shape)
    decoder = model.Decoder(decoder_path, signal_shape)
    critic_x = model.CriticX(critic_x_path, signal_shape)
    critic_z = model.CriticZ(critic_z_path)
    encoder.to(device)
    decoder.to(device)
    critic_x.to(device)
    critic_z.to(device)

    optim_enc = optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_dec = optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_cx = optim.Adam(critic_x.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_cz = optim.Adam(critic_z.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Train
    logging.info('Starting training')
    cx_epoch_loss, cz_epoch_loss, encoder_epoch_loss, decoder_epoch_loss = list(), list(), list(), list()

    for epoch in tqdm(range(args.epoch)):
        logging.info('Epoch {}'.format(epoch))

        cx_nc_loss, cz_nc_loss = list(), list()

        # training data random sampling
        train_dataset = WADIDataset(path_data='./_datasets/WADI/train.npy', sampling_ratio=0.2)
        train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=8, drop_last=True)
        logging.info('Number of train datapoints is {}'.format(len(train_dataset)))
        logging.info('Number of samples in train dataset {}'.format(len(train_dataset)))

        for i in range(1, args.n_critics + 1):
            cx_loss_list, cz_loss_list = list(), list()

            for batch, sample in enumerate(train_loader):
                cx_loss = critic_x_iteration(sample, device, args.batch, args.latent_space_dim, critic_x, decoder,
                                             optim_cx)
                cz_loss = critic_z_iteration(sample, device, args.batch, args.latent_space_dim, critic_z, encoder,
                                             optim_cz)
                cx_loss_list.append(cx_loss)
                cz_loss_list.append(cz_loss)

            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss_list)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss_list)))

        logging.info('Critic training done in epoch {}'.format(epoch))

        encoder_loss, decoder_loss = list(), list()

        for batch, sample in enumerate(train_loader):
            enc_loss = encoder_iteration(sample, device, args.batch, args.latent_space_dim, critic_x, encoder, decoder,
                                         optim_enc)
            dec_loss = decoder_iteration(sample, device, args.batch, args.latent_space_dim, critic_z, encoder, decoder,
                                         optim_dec)
            encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))
        logging.info('Encoder decoder training done in epoch {}'.format(epoch))
        logging.info('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(
            cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), encoder.encoder_path)
            torch.save(decoder.state_dict(), decoder.decoder_path)
            torch.save(critic_x.state_dict(), critic_x.critic_x_path)
            torch.save(critic_z.state_dict(), critic_z.critic_z_path)
            torch.save(optim_enc.state_dict(), encoder_opt_path)
            torch.save(optim_dec.state_dict(), decoder_opt_path)
            torch.save(optim_cx.state_dict(), critic_x_opt_path)
            torch.save(optim_cz.state_dict(), critic_z_opt_path)

    # Test
    test_dataset = WADIDataset(path_data='./_datasets/WADI/test.npy', path_label='./_datasets/WADI/labels.npy')
    test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=8, drop_last=True)
    # anomaly_detection.test(test_loader, encoder, decoder, critic_x, device, args.batch)

    logging.info('Number of samples in test dataset {}'.format(len(test_dataset)))

    reconstruction_error_list = list()
    critic_score_list = list()
    y_true = list()

    for batch, sample in enumerate(test_loader):
        signal = sample['signal'].to(device)
        anomaly = sample['anomaly']

        reconstructed_signal = decoder(encoder(signal))
        reconstructed_signal = torch.squeeze(reconstructed_signal)

        for i in range(0, args.batch):
            x_ = reconstructed_signal[i].detach().cpu().numpy()
            x = signal[i].cpu().numpy()
            y_true.append(int(anomaly[i].detach()))
            reconstruction_error = dtw_reconstruction_error(x, x_)
            reconstruction_error_list.append(reconstruction_error)
        critic_score = torch.squeeze(critic_x(signal))
        critic_score_list.extend(critic_score.detach().cpu().numpy())

        logging.info('test batch:{}, critic_score {}'.format(batch, critic_score))

    reconstruction_error_stats = stats.zscore(reconstruction_error_list)
    critic_score_stats = stats.zscore(critic_score_list)
    anomaly_score = reconstruction_error_stats * critic_score_stats
    y_predict = detect_anomaly(anomaly_score)
    y_predict = prune_false_positive(y_predict, anomaly_score, change_threshold=0.1)
    find_scores(y_true, y_predict)
