import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--tadgan', type=str, default='tadgan')  # add to confirm in console, not used in logic
parser.add_argument('--mode', type=str, default='train')  # train, test, all
parser.add_argument('--device', type=str, default='1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--datasets', type=str, default='wadi') # wadi, swat
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=12)
parser.add_argument('--n_critics', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--latent_space_dim', type=int, default=20)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--sampling_ratio', type=float, default=0.2)  # 0.2


def get_config():
    return parser.parse_args()


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
