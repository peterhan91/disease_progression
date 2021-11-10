import os
import click
import glob
import torch
import dnnlib
import legacy
import PIL.Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pickle5 as pickle
from sklearn.preprocessing import normalize


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def func(elem):
    return int(elem.split('/')[-2].split('_')[-2])

def pred_w(G, dict_name, frame_path, proj_folder, outdir, nn_max=1):
    z_dic = load_obj(dict_name)
    frame = pd.read_csv(frame_path)

    for z in tqdm(frame['name baseline']):
        dfs = frame[frame['name baseline'] == z]
        time = float(dfs['time inter'])
        key = z.split('.')[0]
        pat = key.split('_')[0]
        w = np.load(os.path.join(proj_folder, key, 'projected_w.npz'))['w']
        ws = np.array(list(z_dic.values()))[:,0,0,:]
        difs = normalize(ws, norm='l2') - normalize(w[:,0,:], norm='l2')
        l2 = np.sqrt(np.sum(difs**2, axis=1))
        df = pd.DataFrame(list(z_dic.keys()), columns=['name'])
        df['l2'] = l2
        df = df.sort_values('l2').reset_index(drop=True)

        counter = 0
        vdiffs = np.zeros_like(w)
        pats = []
        pats.append(pat)
        for nn in df['name'].tolist():
            pat_ = nn.split('_')[0]
            lr = nn[-1]
            if not pat_ in pats:
                nns = sorted(glob.glob(proj_folder+pat_+'_*_'+lr+'/projected_w.npz'), key=func)
                time_ = 12*(int(nns[-1].split('/')[-2].split('_')[2][:4])-int(nns[0].split('/')[-2].split('_')[2][:4]))
                nn0 = np.load(nns[0])['w']
                nn1 = np.load(nns[-1])['w']
                vdiffs += nn1 - nn0
                ratio = time_ / time
                vdiffs = vdiffs * ratio
                counter += 1
                pats.append(pat_)
            if counter >= nn_max:
                break
        
        vdiffs = vdiffs/counter
        w_ = vdiffs + w
        ws = torch.from_numpy(w_).cuda()
        synth_image = G.synthesis(ws, noise_mode='const')
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        Path(outdir).mkdir(parents=True, exist_ok=True)
        PIL.Image.fromarray(np.squeeze(synth_image), 'L').save(os.path.join(outdir, 'f_'+z))


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--dict', 'dict_name',      help='dictionary name', default='wdict')
@click.option('--frame', 'frame_path',    help='dataset frame csv name', default=None)
@click.option('--pfolder', 'proj_folder', help='indivdual projection folder', default=None)
@click.option('--outdir', 'outdir',       help='output dir', default='./out')
def run_pred(network_pkl: str, seed: int, dict_name: str, 
            frame_path: str, proj_folder: str, outdir: str):  
    np.random.seed(seed)
    torch.manual_seed(seed)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
    pred_w(G, dict_name, frame_path, proj_folder, outdir)
    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    run_pred()
            
            
            


