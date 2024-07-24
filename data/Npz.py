import numpy as np

npzfile = np.load('/home/exp2/Diffusion_Model/GDSS-master/data/ogbg-molfreesolv_kekulized.npz')
#npzfile = np.load('/home/exp2/Diffusion_Model/GDSS-master/data/zinc250k_kekulized.npz')
#npzfile = np.load('/home/exp2/Diffusion_Model/GDSS-master/data/qm9_kekulized.npz')


for file in npzfile.files:
    print(f"Name: {file}")
    print(f"Shape: {npzfile[file].shape}")
    print(f"Dtype: {npzfile[file].dtype}")
    print(npzfile[file])
    print()

