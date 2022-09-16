import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--datasets', type=str, default='wadi')
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--n_critics', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--latent_space_dim', type=int, default=20)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)


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
