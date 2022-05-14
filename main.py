import math
import itertools
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from tqdm.autonotebook import tqdm, trange
import torch
from dalle_pytorch import DiscreteVAE, DALLE

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

captions = []
images = []
for fn in glob.glob("data/rainbow/*.png"):
    captions.append(os.path.basename(fn).replace(".png", "").split("_"))    
    im = np.array(Image.open(fn))
    images.append(im)
    
images = np.stack(images).astype(float) / 255
images = torch.from_numpy(images).to(torch.float32).permute(0, 3, 1, 2).to(device)

print(len(images), len(captions))


#### Train VAE

def fit(model, opt, criterion, scheduler, train_x, train_y, epochs, batch_size,
        model_file, trainer, n_train_samples=None):
    epoch_loss_train = []

    if n_train_samples is None:
        n_train_samples = train_x.shape[0]

    t = trange(epochs)
    for _ in t:
        rnd_idx = list(range(n_train_samples))
        np.random.shuffle(rnd_idx)
        losses = []
        for batch_idx in range(0, n_train_samples, batch_size):
            model.train()
            opt.zero_grad()
            loss = trainer(model, train_x, train_y, rnd_idx[batch_idx:(batch_idx + batch_size)], criterion)
            loss.backward()
            losses.append(loss.item())
            opt.step()

        epoch_loss_train.append(np.mean(losses))
        scheduler.step()

        t.set_description(f"train: {epoch_loss_train[-1]:.3f}")

    torch.save(model.state_dict(), model_file)
    model.eval()
    return model, epoch_loss_train

vae = DiscreteVAE(
    image_size = images.shape[2],
    num_layers = 3,          # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = 256,       # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = 512,      # codebook dimension
    hidden_dim = 64,         # hidden dimension
    num_resnet_blocks = 2,   # number of resnet blocks
    temperature = 0.9,       # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = False # straight-through for gumbel softmax. unclear if it is better one way or the other
).to(device)

opt = torch.optim.Adam(vae.parameters(), lr=0.001, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.99)

def train_vae_batch(vae, images, _, idx, __):
    loss = vae(images[idx, ...], return_loss = True)
    return loss

vae_model_file = "data/rainbow_vae.model"
if not os.path.exists(vae_model_file):
    vae, loss_history = fit(vae, opt, None, scheduler, images, None, 500, 128, vae_model_file, train_vae_batch)
    plt.plot(loss_history)
    plt.savefig('loss_history.png')
else:
    vae.load_state_dict(torch.load(vae_model_file))

with torch.no_grad():
    all_image_codes = vae.get_codebook_indices(images)
    all_images_decoded = vae.decode(all_image_codes)
    
all_image_codes[np.random.choice(images.shape[0], 10), ...]


### Train Dalle

import itertools

all_words = list(sorted(frozenset(list(itertools.chain.from_iterable(captions)))))
word_tokens = dict(zip(all_words, range(1, len(all_words) + 1)))
caption_tokens = [[word_tokens[w] for w in c] for c in captions]