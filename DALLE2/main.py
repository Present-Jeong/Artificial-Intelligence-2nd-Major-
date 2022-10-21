import math
import itertools
import os
import glob
import gc

from IQA_pytorch import SSIM, utils
from lpips_pytorch import LPIPS, lpips

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from tqdm.autonotebook import tqdm, trange
import torch
import torch.nn as nn
from dalle_pytorch import DiscreteVAE, DALLE
import json

# all_images_decoded = image to image token (after vae)
# all_image_codes = ?
# captions_mask = 텍스트 마스크
# captions_array = 텍스트 어레이값

## Caption
with open('MSCOCO_train_val_Korean.json', 'r',encoding='UTF-8') as f:
    json_data = json.load(f)

org = json.dumps(json_data,ensure_ascii = False)

## device
gc.collect()
torch.cuda.empty_cache()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
#     net = nn.DataParallel(netG, device_ids=list(range(4)))

print(device)

## Dataset, Dataloader
images = []
file_order = []
for fn in tqdm(glob.glob("val2014/*.jpg")[:1000],total=len(glob.glob('val2014/*.jpg')[:100])):
    file_order.append(os.path.basename(fn))
    fn2 = Image.open(fn)
    fn2 = fn2.resize((64, 64))
    im = np.array(fn2)
    if im.shape == (64,64,3):
        images.append(im)

captions = []
for i in tqdm(file_order[:1000],total = len(file_order[:1000])):
    for df in range(len(json_data)):
        # print(json_data[df]['file_path'].split('/')[-1])
        if str(json_data[df]['file_path'].split('/')[-1]) == str(i):
            org_cap = json.dumps(json_data[df]['captions'],ensure_ascii = False)
            captions.append(org_cap)

# captions = []
# images = []
# for fn in glob.glob("data/rainbow/*.png"):
#     captions.append(os.path.basename(fn).replace(".png", "").split("_"))    
#     im = np.array(Image.open(fn))
#     images.append(im)
    
images = np.stack(images).astype(float) / 255
images = torch.from_numpy(images).to(torch.float32).permute(0, 3, 1, 2).to(device)

# print(len(captions),len(images))    
# print(captions[100],images[100].shape,file_order[100])

# images = np.stack(images).astype(float) / 255
# images = torch.from_numpy(images).to(torch.float32).permute(0, 3, 1, 2).to(device)

### Functions

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

def train_vae_batch(vae, images, _, idx, __):
    loss = vae(images[idx, ...], return_loss = True)
    return loss

def train_dalle_batch(vae, train_data, _, idx, __):
    text, image_codes, mask = train_data
    loss = dalle(text[idx, ...], image_codes[idx, ...], return_loss=True) # mask=mask[idx, ...]
    return loss

#### Train VAE

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

vae_model_file = "data/MSCOCO_vae.model"
if not os.path.exists(vae_model_file):
    vae, loss_history = fit(vae, opt, None, scheduler, images, None, 100, 1, vae_model_file, train_vae_batch)
    plt.plot(loss_history)
    plt.savefig('loss_history_discreteVAE.png')
    plt.close()
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

longest_caption = max(len(c) for c in captions)
captions_array = np.zeros((len(caption_tokens), longest_caption), dtype=np.int64)
for i in range(len(caption_tokens)):
    captions_array[i, :len(caption_tokens[i])] = caption_tokens[i]
    
captions_array = torch.from_numpy(captions_array).to(device)
captions_mask = captions_array != 0

dalle = DALLE(
    dim = 1024,
    vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = len(word_tokens) + 1,    # vocab size for text
    text_seq_len = longest_caption,         # text sequence length
    depth = 12,                 # should aim to be 64
    heads = 16,                 # attention heads
    dim_head = 64,              # attention head dimension
    attn_dropout = 0.1,         # attention dropout
    ff_dropout = 0.1            # feedforward dropout
).to(device)

opt = torch.optim.Adam(dalle.parameters(), lr=0.001, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.98)

np.random.seed(1)
train_fraction = 0.3
train_idx = np.random.choice(len(captions), int(len(captions) * train_fraction))

dalle_model_file = "data/MSCOCO_dalle.model"
if not os.path.exists(dalle_model_file):
    dalle, loss_history = fit(dalle, opt, None, scheduler,
    (captions_array[train_idx, ...], all_image_codes[train_idx, ...], captions_mask[train_idx, ...]), None, 100, 1, 
    dalle_model_file, train_dalle_batch, n_train_samples=len(train_idx))

    plt.plot(loss_history)
    plt.savefig('loss_history_dalle.png')
else:
    dalle.load_state_dict(torch.load(dalle_model_file))

generated_images = []
for i in trange(0, len(captions), 64):
    generated = dalle.generate_images(captions_array[i:i + 64, ...], temperature=0.00001) # mask=captions_mask[i:i + 128, ...],
    generated_images.append(generated)

generated_images = torch.cat(generated_images, axis=0).cpu().numpy()

idx = np.random.choice(train_idx, 10)
orig_pics = images[idx, ...].cpu().permute(0, 2, 3, 1).reshape(640, 64, 3).permute(1, 0, 2).numpy()

decoded_pics = generated_images[idx, ...].transpose(0, 2, 3, 1).reshape(640, 64, 3).transpose(1, 0, 2)
both = np.concatenate((orig_pics, decoded_pics), axis=0)
plt.imshow(both)
plt.savefig('train_rainbow.png')

plt.close()
plt.imshow(decoded_pics)
plt.savefig('decoded_pics.png')

ssim_batch = []
lpips_batch = []
ssim = []
lpips_list = []

for i in range(10):
    ref = utils.prepare_image(np.array(orig_pics)).to(device)
    dist = utils.prepare_image(np.array(decoded_pics)).to(device)

    model = SSIM(channels=3)
    score = model(dist, ref, as_loss=False)
    print(score)
    ssim_batch.append(score.tolist())

    ## LPIPS
    lpips_loss = lpips(orig_pics, decoded_pics, net_type='alex', version='0.1')[0][0][0][0]
    lpips_batch.append(lpips_loss.tolist())
    
ssim_batch = np.mean(ssim_batch)
ssim.append(ssim_batch.tolist())
ssim.to_csv('SSIM.csv')
ssim_batch.to_csv('SSIM_batch.csv')
lpips_loss.to_csv('LPIPS.csv')
lpips_batch.to_csv('LPIPS_batch.csv')
print(ssim_batch,ssim,lpips_batch)

test_idx = np.ones(len(captions), bool)
test_idx[train_idx] = False
test_idx = np.flatnonzero(test_idx)
idx = np.random.choice(test_idx, 10)
orig_pics = images[idx, ...].cpu().permute(0, 2, 3, 1).reshape(320, 32, 3).permute(1, 0, 2).numpy()
decoded_pics = generated_images[idx, ...].transpose(0, 2, 3, 1).reshape(320, 32, 3).transpose(1, 0, 2)
both = np.concatenate((orig_pics, decoded_pics), axis=0)
plt.imshow(both)
plt.savefig('test_rainbow.png')
