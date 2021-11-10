import os
import glob
import numpy as np
import pickle

basepath = './project_most/'
files = glob.glob(basepath+'*/projected_w.npz')
dicts = dict()


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    for file in files:
        key = file.split('/')[-2]
        data = np.load(file)['w']
        dicts[key] = data

    save_obj(dicts, 'wdict_most')
    
