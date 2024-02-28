import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt 
import os 
import torch.autograd as autograd
from torch.autograd import Variable

import torchvision.transforms.functional as TF
import random

def transform_CT(img, mu_w=0.0193):
    return (img-mu_w)/mu_w*1000

def inv_transform_CT(img, mu_w=0.0193):
    return img*mu_w/1000+mu_w

def map_to_zero_one(X, return_vals=False):
    min_val = X.min(-1)[0].min(-1)[0]
    max_val = X.max(-1)[0].max(-1)[0]
    X = (X-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
    if return_vals:
        return X, min_val, max_val
    else:
        return X
    
def map_to_zero_one_alt(X, min_v = None, max_v = None, return_vals=False):
    if min_v == None:
        min_val = X.min()
        max_val = X.max()
    else:
        min_val = min_v
        max_val = max_v
    X = (X-min_val)/(max_val-min_val)
    if return_vals:
        return X, min_val, max_val
    else:
        return X
    
def inv_map_to_zero_one(X, min_val, max_val,dataset):
    n_mat = X.size(1)
    if dataset in ['pred','obs']:
        X = X[:,0:n_mat,:,:]*(max_val[:,0:n_mat,None,None]-min_val[:,0:n_mat,None,None])+min_val[:,0:n_mat,None,None]
    else:
        X = X[:,0:n_mat,:,:]*(max_val[:,n_mat:n_mat*2,None,None]-min_val[:,n_mat:n_mat*2,None,None])+min_val[:,n_mat:n_mat*2,None,None]     
    return X

def inv_map_to_zero_one_alt(X, min_val, max_val):
    return X*(max_val-min_val)+min_val

def inv_map_to_zero_one_mono(X, min_val, max_val):
    n_mat = X.size(1)
    return X*(max_val[:,:,None,None]-min_val[:,:,None,None])+min_val[:,:,None,None]    

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def prepare_dataloaders(train_file, valid_file,  batch_sz, standardize):
    train_data = torch.load(train_file + '.pt')
    valid_data = torch.load(valid_file + '.pt')
    
    train_std = None 
    if standardize:
        #https://discuss.pytorch.org/t/standardization-of-data/16965/4
        train_std = train_data.std((0,2,3), keepdim=True)
        train_data /= train_std
        valid_data /= train_std 
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_sz, shuffle=True)
    
    return train_loader, valid_loader, train_std 

# based on function with same name from the 2017 ODL workshop
def show_image_matrix(images, titles, figsize = 10, save_path = None, **kwargs):
    n_samples, n_mat = images[0].size()[0:2]
    n_cols = len(images)
    
    fig, axs = plt.subplots(n_mat*n_samples, n_cols, sharex=True, sharey=True,
        figsize=(n_cols*figsize, figsize*n_mat*n_samples))
    idx = 0
    for i in range(n_samples):
        for mat in range(n_mat):
            col = axs[idx]
            idx += 1
            for title, img, ax in zip(titles, images, col):
                ax.set_title(title)
                plot = ax.imshow(img[i,mat,:,:], **kwargs)
                ax.set_axis_off()
                fig.colorbar(plot, ax = ax)
    if save_path is not None:
        plt.savefig(save_path+'.png')
    plt.show()

# https://stackoverflow.com/questions/13583153/how-to-zoomed-a-portion-of-image-and-insert-in-the-same-plot-in-matplotlib
# https://github.com/matplotlib/matplotlib/issues/12323/
def plot_insert(img, save=None, **kwargs):
    fig, ax = plt.subplots(figsize=[10, 10])

    #plot = ax.imshow(img, **kwargs)
    ax.imshow(img, **kwargs)

    # inset axes....
    axins = ax.inset_axes([0.55, 0.55, 0.45, 0.45]) #x_0,y_0,h,w
    axins.imshow(img, **kwargs)
    
    # sub region of the original image
    img_size = img.size()[-1]
    x1, x2, y1, y2 = (img_size//2-32), (img_size//2+32), (img_size//2-32), (img_size//2+32)
    #x1, x2, y1, y2 = 266, 316, 316, 266
    #x1, x2, y1, y2 = 241, 291, 291, 241
    #x1, x2, y1, y2 = 191, 291, 291, 191

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    #ax.indicate_inset_zoom(axins, edgecolor="black")
    rectpatch, connects=ax.indicate_inset_zoom(axins,edgecolor="black")
    connects[0].set_visible(False)
    connects[1].set_visible(False)
    connects[2].set_visible(False)
    connects[3].set_visible(False)
    
    #fig.colorbar(plot, ax = ax)
    
    plt.show()
    if save is not None:
        fig.savefig(save, dpi = 120)

def get_mono(data):
    mu_soft = 0.0176 # 0.0203
    mu_bone =  0.0395 # 0.0492
    return data[:,0:1,:,:]*mu_soft+data[:,1:,:,:]*mu_bone

#https://github.com/pytorch/pytorch/issues/7415
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def get_gradient_penalty(critic, real_data, fake_data, device):
    # sample epsilon ~ U[0,1] for interpolation between real and fake 
    epsilon = torch.rand(len(real_data), 1, 1, 1, device=device, requires_grad=True)
    # interpolate real and fake 
    interpolates = epsilon*real_data+(1-epsilon)*fake_data 
    # pass interpolates through critic 
    c_interpolates = critic(interpolates)
    # compute gradient wrt to inputs 
    gradients = autograd.grad(
        outputs = c_interpolates, 
        inputs = interpolates,
        grad_outputs = torch.ones_like(c_interpolates),
        create_graph = True,
        retain_graph = True, 
        only_inputs = True 
        )[0]
    # flatten gradients 
    gradients = gradients.view(gradients.size(0),-1)
    gradient_penalty = torch.mean((gradients.norm(2,dim=1)-1)**2)
    return gradient_penalty 
