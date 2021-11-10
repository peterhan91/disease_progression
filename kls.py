import glob
import fnmatch
import numpy as np
import pandas as pd


most_df = pd.read_csv('./label/MOSTV01235XRAY.txt', sep='\t')
most_df = most_df.filter(items=['MOSTID','ID',
                                'V0XRKL','V1XRKL','V2XRKL','V3XRKL','V5XRKL',
                                'V0XLKL','V1XLKL','V2XLKL','V3XLKL','V5XLKL'])

imgs_ = sorted(glob.glob('../Xray/dataset_most/imgs/*.png'))
files = list(glob.glob('../MOST/*/*/*/*/*/*'))

kls = []
print(len(imgs_))
imgs = [img.split('/')[-1] for img in imgs_]
for n, img in enumerate(imgs):
    pat = img.split('_')[0]
    lr = img.split('_')[-1].split('.')[0].upper()
    visit = img.split('_')[1]
    try:
        pattern = '*'+img.split('_')[2]
        path = fnmatch.filter(files, pattern)[0]
        month = path.split('/')[-4]

        df = most_df[most_df['MOSTID']==pat]
        kl = df[month+'X'+lr+'KL'].tolist()[0]
        if kl in ['P','S','X','Z','8','9']:
            kls.append(np.nan)
        else:
            kls.append(float(kl))
    except:
        kls.append(np.nan)


d = {'img name': imgs, 'kls': kls}
mostkl = pd.DataFrame(data=d)
mostkl.to_csv('./label/most_kls.csv', index=False)
