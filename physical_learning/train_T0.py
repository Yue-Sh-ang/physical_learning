from IPython.display import HTML

import numpy as np
import matplotlib.pyplot as plt

from packing_utils import *
from allosteric_utils import *
from plot_imports import *
from lammps_utils import *

import sys
if (len(sys.argv) != 4):
    print('Usage: train_T0.py contact seed es')
    sys.exit()

contact = float(sys.argv[1])
seed = int(sys.argv[2])
es = float(sys.argv[3])
n=48#default number of particles
temp=0

net = Packing(n, rfac=0.8, seed=seed)
net.params['contact'] = contact # change the default contact repulsion
net.generate()

allo=Allosteric(net.graph, dim=2)
allo.add_targets(seed=0,plot=False)
allo.add_sources(seed=0,plot=False)    #alpha = float(re.search(r'alpha([\d.]+)k', folder).group(1))
l2 = np.mean([edge[2]['length']**2 for edge in allo.graph.edges(data=True)])
alpha = 1e-4/l2
eta=1
sol = allo.solve(duration=1e7,alpha=alpha,eta=eta, frames=200, T=0,train=2, applied_args=(es, es, 100))

path= f'/data2/shared/yueshang/lammps_docs/check_lowT_mech/contact{contact:.3f}/seed{seed}/T{temp:.3g}/es{es:.1f}/'
os.makedirs(path, exist_ok=True)
allo.save(path + 'allo.txt')
traj=allo.traj
np.save(path + 'traj.npy', traj)
