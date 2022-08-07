import os
import glob
import torch
import imageio
import argparse
import PIL.Image
import pandas as pd
import numpy as np
import pickle5 as pickle
from pathlib import Path
from sklearn.preprocessing import normalize
import dnnlib
import legacy

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def func(elem):
    return int(elem.split('/')[-2].split('_')[-2])

################# nearest neighbor search ################# 
def get_nn(w, ws):
    difs = normalize(ws, norm='l2') - normalize(w[:,0,:], norm='l2')
    l2 = np.sqrt(np.sum(difs**2, axis=1))
    df = pd.DataFrame(list(z_dic.keys()), columns=['name'])
    df['l2'] = l2
    df = df.sort_values('l2').reset_index(drop=True)
    return df

################# predict single time (next) X-ray ################# 
def get_single_pred(args, w, dictionary):
    df = get_nn(w, dictionary)
    time = float(df['time inter'])
    # print(df.head())
    pats = []
    pats.append(pat)
    for nn in df['name'].tolist():
        pat_ = nn.split('_')[0]
        lr = nn[-1]
        if not pat_ in pats:
            nns = sorted(glob.glob(args.proj_folder+pat_+'_*_'+lr+'/projected_w.npz'), key=func)
            # print('neighbor', nn, 'length', len(nns))
            names = [x.split('/')[-2] for x in nns]
            idx = names.index(nn)
            time_nn = df[df['name'] == nn]['time inter'].tolist()[0]
            nn0 = np.load(nns[idx])['w']
            try: 
                nn1 = np.load(nns[idx+1])['w']
                ratio = time / time_nn
                vdiffs = (nn1 - nn0) * ratio
                w_ = vdiffs + w
                ws = torch.from_numpy(w_).cuda()
                synth_image = G.synthesis(ws, noise_mode='const')
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                # print('single prediction finished!')
                return synth_image, vdiffs
            except:
                continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=303, help='random seeds for experiment reproduction')
    parser.add_argument('--dict_path', type=str, default='../../stylegan2-ada-pytorch/wdict.pkl', 
                                                    help='directory of the latent mapping dictionary')
    parser.add_argument('--network_pkl', type=str, default='../../stylegan2-ada-pytorch/training-runs/00001-oai-auto2/network-snapshot-011400.pkl', 
                                                    help='network saved checkpoint path')
    parser.add_argument('--csv_path', type=str, default='../label/oai_ptest.csv', 
                                                    help='path to OAI csv file')
    parser.add_argument('--proj_folder', type=str, default='../label/oai_ptest.csv', 
                                                    help='latent projection folder containing all training images')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    
    z_dic = load_obj(args.dict_path)
    print('Loading networks from "%s"...' % args.network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(args.network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
    df = pd.read_csv(args.csv_path)

    for z in df['name baseline'].tolist():
        key = z.split('.')[0]
        pat = key.split('_')[0]
        lr = key[-1]
        dictionary = np.array(list(z_dic.values()))[:,0,0,:]
        ws = sorted(glob.glob(args.proj_folder+pat+'_*_'+lr+'/projected_w.npz'), key=func)
        names = [x.split('/')[-2] for x in ws]
        print('total data points', len(ws))
        w0 = np.load(os.path.join(args.proj_folder, key, 'projected_w.npz'))['w']
        vdif = 0
        
        outdir = os.path.join('multipoint/prog+', key)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        
        for n in range(len(ws)):
            if n == 0:
                img_g = imageio.imread(os.path.join(args.proj_folder, names[n], 'proj.png'))
                img_r = imageio.imread(os.path.join(args.proj_folder, names[n], 'target.png'))
            else:
                w = w0 + vdif
                img_g, dif = get_single_pred(args, w, dictionary)
                img_r = imageio.imread(os.path.join(args.proj_folder, names[n], 'target.png'))
                vdif += dif
            img_t = np.concatenate((img_r, np.squeeze(img_g)), axis=0)
            PIL.Image.fromarray(np.squeeze(img_t), 'L').save(os.path.join(outdir, names[n] + '.png')) 