import os
import click
import glob
from numpy.lib.polynomial import RankWarning
import torch
import imageio
import PIL.Image
import pandas as pd
import numpy as np
import pickle5 as pickle
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from skimage import exposure
from skimage.transform import resize
from sklearn.preprocessing import normalize
import dnnlib
import legacy

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def func(elem):
    return int(elem.split('/')[-2].split('_')[-2])

def plot(test_folder):
    if test_folder.split('/')[-3].split('_')[0] == 'nonprogressive':
        surfix = 'n'
    elif test_folder.split('/')[-3].split('_')[0] == 'progressive':
        surfix = 'p'
    outdir = '../OAI_Xray/OAIMOST_test/result_' + surfix
    Path(outdir).mkdir(parents=True, exist_ok=True)
    dir = '../OAI_Xray/OAIMOST_test/' + test_folder.split('/')[-3]

    for target in glob.glob(os.path.join(dir, '*/baseline.png')):
        try:
            path = os.path.dirname(target)
            pat = target.split('/')[-2]
            lr = pat[-1]
            init = imageio.imread(target)
            reco = imageio.imread(os.path.join(path, 'baseline_re.png'))
            try:
                pred = imageio.imread(os.path.join(path, 'pred.png'))
            except FileNotFoundError:
                continue
            late = imageio.imread(os.path.join(path, 'late.png'))
            upper = np.concatenate((init, reco), axis=1)
            lower = np.concatenate((late, pred), axis=1)
            total = np.concatenate((upper, lower), axis=0)
            imageio.imwrite(os.path.join(outdir, pat+'.png'), total)
        except IndexError:
            print(pat)
            pass

def pred_w(G, if_comp=True):
    if if_comp:
        oai_dict = load_obj('wdict')
        most_dict = load_obj('wdict_most')
        z_dic = {**oai_dict, **most_dict}
        frame_oai = pd.read_csv('../OAI_distinguish/label/oai_ptest.csv')
        frame_most = pd.read_csv('../OAI_distinguish/label/most_ptest.csv')
        frame = frame_most[frame_most['fast prog']==1]

        for z in tqdm(frame['name baseline']):
            dfs = frame[frame['name baseline'] == z]
            time = float(dfs['time inter'])
            key = z.split('.')[0]
            pat = key.split('_')[0]
            w = np.load(os.path.join('./project_most', key, 'projected_w.npz'))['w']
            ws = np.array(list(z_dic.values()))[:,0,0,:]
            difs = normalize(ws, norm='l2') - normalize(w[:,0,:], norm='l2')
            l2 = np.sqrt(np.sum(difs**2, axis=1))
            df = pd.DataFrame(list(z_dic.keys()), columns=['name'])
            df['l2'] = l2
            # df = df.sample(frac=0.01, random_state=10)
            df = df.sort_values('l2').reset_index(drop=True)

            counter = 0
            vdiffs = np.zeros_like(w)
            pats = []
            pats.append(pat)
            for nn in df['name'].tolist():
                pat_ = nn.split('_')[0]
                lr = nn[-1]
                if not pat_ in pats:
                    if nn in oai_dict:
                        nns = sorted(glob.glob('./project/'+pat_+'_*_'+lr+'/projected_w.npz'), key=func)
                        time_ = 12*(int(nns[-1].split('/')[-2].split('_')[2][:4])-int(nns[0].split('/')[-2].split('_')[2][:4]))
                    else:
                        nns = sorted(glob.glob('./project_most/'+pat_+'_*_'+lr+'/projected_w.npz'), key=func)
                        most_df = frame_most[frame_most['ID'] == pat_]
                        side = 1 if lr == 'r' else 2
                        most_df = most_df[most_df['SIDE'] == side]
                        if len(most_df)>0:
                            time_ = float(most_df['time inter'])
                        else:
                            time_ = time
                    nn0 = np.load(nns[0])['w']
                    nn1 = np.load(nns[-1])['w']
                    vdiffs += nn1 - nn0
                    # ratio = 1 if time_==0.0 else time / time_
                    # vdiffs = vdiffs * ratio
                    counter += 1
                    pats.append(pat_)
                if counter > 0:
                    break
            
            vdiffs = vdiffs/counter
            w_ = vdiffs + w
            ws = torch.from_numpy(w_).cuda()
            synth_image = G.synthesis(ws, noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            outdir = '../OAI_Xray/MOST_test/pred_prog'
            Path(outdir).mkdir(parents=True, exist_ok=True)
            PIL.Image.fromarray(np.squeeze(synth_image), 'L').save(os.path.join(outdir, 'f_'+z))


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--subsample', 'samplerate', help='dict subsample rate', default=None)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
def run_pred(network_pkl: str, samplerate: float, seed: int,):  
    np.random.seed(seed)
    torch.manual_seed(seed)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    pred_w(G, if_comp=True)
    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    run_pred()
            
            
            


