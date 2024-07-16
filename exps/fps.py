import torch

import os
import time
from glob import glob
from tqdm import tqdm
from statistics import mean
import pandas as pd



# models = sorted(glob('ablation/SEG_RES/*.pt'), key=lambda x: int(os.path.basename(x).split('-')[3].split('x')[0]))
models = sorted(glob('*.pt'), key=lambda x: int(os.path.basename(x).split('-')[3].split('x')[0]))


N = 10000

results = []
for model in models:
    res = os.path.basename(model).split('-')[3]
    H,W = res.split('x')
    mo = torch.jit.load(model)
    in_img = torch.rand((1,3,int(H),int(W)), dtype=torch.float).to('cuda')
    
    times = []
    for _ in tqdm(range(N)):
        t1 = time.time()
        out = mo(in_img)
        t2 = time.time()
        times.append(t2-t1)
    
    results.append({
        'model': os.path.splitext(os.path.basename(model))[0],
        'time': mean(times),
        'FPS': 1/(mean(times))
    })
    
results = pd.DataFrame(results)
results.model = results.apply(lambda x: x.model.split('-')[-3], axis=1)
# results = results.sort_values(by='model', key = lambda col: ) for x in col])
results.to_csv('seg-res-FPS.csv', index=False)
print(results)