"""
Network developed by Dennis Hein. Changes made to take 2 input channels and one output channel (2 VMI's -> 1 SPR map).
"""

# import packages
import argparse 
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm.auto import tqdm
from torch import nn
from torch import optim
from torchvision import transforms 
from IPython.display import clear_output
import pytorch_ssim
import math 
import time 

# user defined
import models 
import utils
#import sr3
import sr3_alt
#import sr3_alt_alt
import sr3_alt_2VMI

rotation_transform = utils.MyRotationTransform(angles=[0, 90, 180, 270])
hflip_transform = transforms.RandomHorizontalFlip()

def train_loop(data, model, loss_fn, optimizer, lambda_1, lambda_2, vgg, vgg_alt, batch_sz, patch_sz, n_patches, device, display_interval,aug_rotate,aug_flip):
    
    n, n_mat, h, w = data.size()
    #n_mat //= 2
    #n_mat = math.ceil(n_mat/2)
    n_mat = n_mat-1 #always one SPR
    n_batches = math.ceil(n/batch_sz)*n_patches
    # set up loss 
    loss_mse = nn.MSELoss() 
    loss_l1 = nn.L1Loss()
    model.train() 
    
    train_loss = 0
    i = 0 
    while i < n_batches:         
        start = time.time()
        # sample batch 
        data_batch = data[torch.randperm(n-batch_sz)[0:batch_sz],:,:,:].to(device)
        if patch_sz is None:
            for j in range(0,batch_sz):
                if aug_flip:
                    data_batch[j,:,:,:] = hflip_transform(data_batch[j,:,:,:])
                if aug_rotate:
                    data_batch[j,:,:,:] = rotation_transform(data_batch[j,:,:,:])
            obs = data_batch[:,0:n_mat,:,:]
            truth = data_batch[:,n_mat:n_mat*2,:,:]
        else:
            h_start = np.random.choice(np.array(range(0,h-patch_sz)), batch_sz)
            w_start = np.random.choice(np.array(range(0,w-patch_sz)), batch_sz)
            obs = torch.zeros((batch_sz,n_mat,patch_sz,patch_sz)).to(device)
            truth = torch.zeros((batch_sz,n_mat,patch_sz,patch_sz)).to(device)
            for j in range(0,batch_sz):
                idx_h = torch.tensor(range(h_start[j], h_start[j]+patch_sz)).to(device)
                idx_w = torch.tensor(range(w_start[j], w_start[j]+patch_sz)).to(device)
                if aug_flip:
                    data_batch[j,:,:,:] = hflip_transform(data_batch[j,:,:,:])
                if aug_rotate:
                    data_batch[j,:,:,:] = rotation_transform(data_batch[j,:,:,:])
                obs[j,:,:,:] = data_batch[j,0:n_mat,:,:].index_select(1, idx_h).index_select(2, idx_w) 
                truth[j,:,:,:] = data_batch[j,n_mat:n_mat*2,:,:].index_select(1, idx_h).index_select(2, idx_w)
                                                                 
                
        # compute prediction and loss
        pred = model(obs)
        if loss_fn == 'mse':
            loss = loss_mse(pred, truth)
        elif loss_fn == 'mse_alt':
            loss = loss_mse(torch.mean(pred), torch.mean(truth))
        elif loss_fn == 'mse_l1':
            loss = lambda_1*loss_mse(pred, truth) + lambda_2*loss_l1(pred,truth)
        elif loss_fn == 'l1':
            loss = loss_l1(pred, truth)
        elif loss_fn == 'vgg16' or loss_fn == 'vgg19':
            loss = loss_mse(vgg(pred), vgg(truth))
        elif loss_fn == 'vgg16_alt':
            loss = lambda_1*loss_mse(vgg(pred),vgg(truth))+lambda_2*loss_mse(vgg_alt(torch.mean(pred,1,keepdim=True)),vgg_alt(torch.mean(truth,1,keepdim=True)))
        elif loss_fn == 'vgg16_mse' or loss_fn == 'vgg19_mse':
            loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_mse(pred,truth)
        elif loss_fn == 'vgg16_mse_alt' or loss_fn == 'vgg19_mse_alt':
            loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_mse(torch.mean(pred),torch.mean(truth))
        elif loss_fn == 'vgg16_l1' or loss_fn == 'vgg19_l1':
            loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_l1(pred,truth)
        elif loss_fn == 'vgg16_l1_alt' or loss_fn =='vgg19_l1_alt':
            loss = lambda_1*loss_mse(vgg(utils.get_mono(pred)),vgg(utils.get_mono(truth)))+lambda_2*loss_l1(pred,truth)
        else:
            raise RuntimeError('Please provide a supported loss function')

           
        # backpropagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() / n_batches
        end = time.time()
        
        # print progress 
        if i % display_interval == 0:
            print('Batch: [{}/{}] Loss: {:.7f} Time per iter. {:.3f}'.format(i, n_batches, loss.item(),end-start))
        
        i +=1
    
    return train_loss

def test_loop(data, model, loss_fn, lambda_1, lambda_2, vgg, vgg_alt, n_samples, fig_sz, batch_sz, patch_sz, n_patches, device, aug_rotate, aug_flip, standardize, standardize_alt, min_val, max_val,ct_trans):
    
    n, n_mat, h, w = data.size()
    #n_mat //= 2
    #n_mat = math.ceil(n_mat / 2)
    n_mat = n_mat-1
    n_batches = math.ceil(n/batch_sz)*n_patches
        
    # set up loss
    loss_mse = nn.MSELoss() 
    loss_l1 = nn.L1Loss()
    loss_ssim = pytorch_ssim.SSIM(window_size = 11)
    
    mean_loss = 0 
    mean_psnr = 0
    mean_ssim = 0

    model.eval()
    
    i = 0
    with torch.no_grad():
        while i < n_batches:         
            start = time.time()
            # sample batch 
            if standardize:
                data_batch, val_min, val_max = utils.map_to_zero_one(data[torch.randperm(n-batch_sz)[0:batch_sz],:,:,:].to(device),return_vals=True)
            elif standardize_alt:
                data_batch = utils.map_to_zero_one_alt(data[torch.randperm(n-batch_sz)[0:batch_sz],:,:,:].to(device), min_val, max_val)
            else:
                data_batch = data[torch.randperm(n-batch_sz)[0:batch_sz],:,:,:].to(device)
            if patch_sz is None:
                for j in range(0,batch_sz):
                    if aug_flip:
                        data_batch[j,:,:,:] = hflip_transform(data_batch[j,:,:,:])
                    if aug_rotate:
                        data_batch[j,:,:,:] = rotation_transform(data_batch[j,:,:,:])
                    obs = data_batch[:,0:n_mat,:,:]
                    truth = data_batch[:,n_mat:n_mat*2,:,:]
            else:
                h_start = np.random.choice(np.array(range(0,h-patch_sz)), batch_sz)
                w_start = np.random.choice(np.array(range(0,w-patch_sz)), batch_sz)
                obs = torch.zeros((batch_sz,n_mat,patch_sz,patch_sz)).to(device)
                truth = torch.zeros((batch_sz,n_mat,patch_sz,patch_sz)).to(device)
                for j in range(0,batch_sz):
                    idx_h = torch.tensor(range(h_start[j], h_start[j]+patch_sz)).to(device)
                    idx_w = torch.tensor(range(w_start[j], w_start[j]+patch_sz)).to(device)
                    if aug_flip:
                        data_batch[j,:,:,:] = hflip_transform(data_batch[j,:,:,:])
                    if aug_rotate:
                        data_batch[j,:,:,:] = rotation_transform(data_batch[j,:,:,:])
                    obs[j,:,:,:] = data_batch[j,0:n_mat,:,:].index_select(1, idx_h).index_select(2, idx_w) 
                    truth[j,:,:,:] = data_batch[j,n_mat:n_mat*2,:,:].index_select(1, idx_h).index_select(2, idx_w)
            
            # compute prediction and loss
            pred = model(obs)
            if loss_fn == 'mse':
                loss = loss_mse(pred, truth)
            elif loss_fn == 'mse_alt':
                loss = loss_mse(torch.mean(pred), torch.mean(truth))
            elif loss_fn == 'mse_l1':
                loss = lambda_1*loss_mse(pred, truth) + lambda_2*loss_l1(pred,truth)
            elif loss_fn == 'l1':
                loss = loss_l1(pred, truth)
            elif loss_fn == 'vgg16' or loss_fn == 'vgg19':
                loss = loss_mse(vgg(pred), vgg(truth))
            elif loss_fn == 'vgg16_alt':
                loss = lambda_1*loss_mse(vgg(pred),vgg(truth))+lambda_2*loss_mse(vgg_alt(torch.mean(pred,1,keepdim=True)),vgg_alt(torch.mean(truth,1,keepdim=True)))
            elif loss_fn == 'vgg16_mse' or loss_fn == 'vgg19_mse':
                loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_mse(pred,truth)
            elif loss_fn == 'vgg16_mse_alt' or loss_fn == 'vgg19_mse_alt':
                loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_mse(torch.mean(pred),torch.mean(truth))
            elif loss_fn == 'vgg16_l1' or loss_fn == 'vgg19_l1':
                loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_l1(pred,truth)
            elif loss_fn == 'vgg16_l1_alt' or loss_fn =='vgg19_l1_alt':
                loss = lambda_1*loss_mse(vgg(utils.get_mono(pred)),vgg(utils.get_mono(truth)))+lambda_2*loss_l1(pred,truth)
            else:
                raise RuntimeError('Please provide a supported loss function')
            # other performance metrics 
            psnr = 10 * np.log10((torch.max(truth).item()**2) / loss_mse(pred,truth).item()) 
            ssim = loss_ssim(pred, truth)
            
            # add to sum    
            mean_loss += loss.item()
            mean_psnr += psnr 
            mean_ssim += ssim.item()
            
            if batch_sz < n_samples:
                n_samples = batch_sz 
            
            if standardize:
                pred = utils.inv_map_to_zero_one(pred,val_min,val_max,'pred')
                obs = utils.inv_map_to_zero_one(obs,val_min,val_max,'obs')
                truth = utils.inv_map_to_zero_one(truth,val_min,val_max,'truth')
            if standardize_alt:
                pred = utils.inv_map_to_zero_one_alt(pred,min_val,max_val)
                obs = utils.inv_map_to_zero_one_alt(obs,min_val,max_val)
                truth = utils.inv_map_to_zero_one_alt(truth,min_val,max_val)
           
            # plot example output 
            if i < n_samples:
                 if n_mat == 1:
                     if ct_trans:
                         fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                             figsize=(2*fig_sz, fig_sz*3))
                         axs[0].imshow(utils.transform_CT(obs[i,0,:,:].cpu()), cmap='bone',clim=[0,70])
                         axs[1].imshow(utils.transform_CT(pred[i,0,:,:].detach().cpu()), cmap='bone',clim=[0,70])
                         axs[2].imshow(utils.transform_CT(truth[i,0,:,:].detach().cpu()), cmap='bone',clim=[0,70])
                         plt.show()
                  #   else:
                   #      fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                   #           figsize=(2*fig_sz, fig_sz*3))
                   #      axs[0].imshow((obs[i,0,:,:].cpu()), cmap='bone',clim=[0,70])
                   #      #axs[1].imshow((pred[i,0,:,:].detach().cpu()), cmap='bone',clim=[0,70])
                   #      #axs[2].imshow((truth[i,0,:,:].detach().cpu()), cmap='bone',clim=[0,70])
                   #      axs[1].imshow((pred[i,0,:,:].detach().cpu()), cmap = 'gray', clim=[0, 2])
                   #      axs[2].imshow((truth[i,0,:,:].detach().cpu()), cmap= 'gray', clim=[0, 2])
                   #      plt.show()
                
#                 else:
             #        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True,
             #                                figsize=(2 * fig_sz, fig_sz * 3))
             ##        axs[0,0].imshow((obs[i, 0, :, :].cpu()), cmap='bone', clim=[0, 70])
              #       axs[1,0].imshow((obs[i, 1, :, :].cpu()), cmap='bone', clim=[0, 70])
              #       # axs[1].imshow((pred[i,0,:,:].detach().cpu()), cmap='bone',clim=[0,70])
              #       # axs[2].imshow((truth[i,0,:,:].detach().cpu()), cmap='bone',clim=[0,70])
              #       axs[0,1].imshow((pred[i, 0, :, :].detach().cpu()), cmap='gray', clim=[0, 2])
              #       #axs[1,1].imshow((pred[i, 1, :, :].detach().cpu()), cmap='gray', clim=[0, 2])

                #     axs[0,2].imshow((truth[i, 0, :, :].detach().cpu()), cmap='gray', clim=[0, 2])
                     #axs[1,2].imshow((truth[i, 1, :, :].detach().cpu()), cmap='gray', clim=[0, 2])

#                     plt.show()


            i +=1

    # get averages
    test_loss = mean_loss / n_batches
    mean_psnr /= n_batches
    mean_ssim /= n_batches
    
    print('Test results: \n ---------------------------')
    print('Mean test loss: {:.7f}'.format(test_loss))
    print('Mean PSNR: {:.7f}'.format(mean_psnr))
    print('Mean SSIM: {:.7f}'.format(mean_ssim))
    return torch.tensor([test_loss])

def main(args):
    # set up data 
    print('Setting up data...')
    train_data = torch.load(args.train+'.pt') 
    val_data = torch.load(args.valid+'.pt')
    print('Data set up done!')
    
    ct_trans = False
    if 'invmm' in args.train.split('_'):
        ct_trans = True
    
    if args.standardize==True:
        train_data = utils.map_to_zero_one(train_data)

    min_val = None
    max_val = None
    if args.standardize_alt==True:
        train_data, min_val, max_val = utils.map_to_zero_one_alt(train_data, return_vals=True)
        torch.save(torch.cat((min_val.unsqueeze(0),max_val.unsqueeze(0)),dim=0),args.train+'_minmax.pt')
    
    if args.ch_mult == 'celebahq':
        ch_mult = (1, 2, 4, 8, 16, 32, 32, 32)
    elif args.ch_mult == 'celebahq256':
        ch_mult = (1, 1, 2, 2, 2, 2, 2)
    
    device = 'cuda:'+str(args.device)    
        
    # set up model
    #n_mat = train_data.size(1)//2
    #n_mat = math.ceil(train_data.size(1) / 2)
    n_mat = train_data.size(1)-1
    if args.patch_sz is not None:
        img_sz = args.patch_sz
    else:
        img_sz = train_data.size(2)
    if args.net == 'resnet':
        model = models.iterative_ResNet(args.n_iter, n_mat, args.n_channels).to(device)
        save = args.net+'_'+str(args.n_iter)+'_'+str(args.n_channels) 
    elif args.net == 'unet':
        model = models.UNet(n_mat,args.init_features,norm=args.batch_norm).to(device)
        save = args.net+'_'+str(args.init_features)
    elif args.net == 'unet_alt':
        model = models.UNet_alt(n_mat,args.init_features,norm=args.batch_norm,skip=args.skip_connection, pre=args.pre_activation).to(device)
        save = args.net+'_'+str(args.init_features)
    elif args.net == 'yang':
        model = models.Generator_yang(n_mat,args.init_features).to(device)
        save = args.net+'_'+str(args.init_features)
    elif args.net == 'cycle':
        model = models.Generator_cycle(n_mat,n_mat,args.init_features).to(device)
        save = args.net+'_'+str(args.init_features)
    elif args.net == 'cycle_alt':
        model = models.Generator_cycle_alt(n_mat,n_mat,args.init_features).to(device)
        save = args.net+'_'+str(args.init_features)
    elif args.net == 'sr3':
        model = sr3.NCSNpp(num_channels=n_mat, image_size = img_sz, nf=args.init_features, num_resblocks = args.num_resblocks, skip=args.skip_connection).to(device)
        save = args.net+'_'+str(args.init_features) +'_'+ str(args.num_resblocks)
    elif args.net == 'sr3_alt':
        model = sr3_alt.NCSNpp(num_channels=n_mat, image_size = img_sz, nf=args.init_features, num_resblocks = args.num_resblocks, skip=args.skip_connection, ch_mult=ch_mult).to(device)
        save = args.net+'_'+str(args.init_features) +'_'+ str(args.num_resblocks) +'_'+ args.ch_mult
    elif args.net == 'sr3_alt_alt':
        model = sr3_alt_alt.NCSNpp(num_channels=n_mat, image_size = img_sz, nf=args.init_features, num_resblocks = args.num_resblocks, skip=args.skip_connection, ch_mult=ch_mult).to(device)
        save = args.net+'_'+str(args.init_features) +'_'+ str(args.num_resblocks) +'_'+ args.ch_mult
    elif args.net == 'sr3_alt_2VMI':
        model = sr3_alt_2VMI.NCSNpp(num_channels=n_mat, output_channels=1, image_size = img_sz, nf=args.init_features, num_resblocks = args.num_resblocks, skip=args.skip_connection, ch_mult=ch_mult).to(device)
        save = args.net+'_'+str(args.init_features) +'_'+ str(args.num_resblocks) +'_'+ args.ch_mult + 'new_for_check14'
    else:
        raise RuntimeError('Please provide a supported model')

    # to get sense of complexity 
    print('Total number of parameters:',
      sum(param.numel() for param in model.parameters()))
    
    # set up perceptual loss 
    vgg = None
    vgg_alt = None
    if args.loss_fn=='vgg16_l1_alt' or args.loss_fn =='vgg16_alt': 
        vgg = models.VGG_Feature_Extractor_16(layer=args.layer, n_mat=1,requires_grad=False).to(device)
        vgg_alt = models.VGG_Feature_Extractor_16(layer=args.layer, n_mat=1,requires_grad=False).to(device)
    elif args.loss_fn=='vgg19_l1_alt' or args.loss_fn =='vgg19_alt': 
        vgg = models.VGG_Feature_Extractor_19(layer=args.layer, n_mat=1,requires_grad=False).to(device)
    elif args.loss_fn.split('_')[0] == 'vgg16':
        vgg = models.VGG_Feature_Extractor_16(layer=args.layer,n_mat=1,requires_grad=False).to(device)
    elif args.loss_fn.split('_')[0] == 'vgg19':
        vgg = models.VGG_Feature_Extractor_19(layer=args.layer,n_mat=1,requires_grad=False).to(device)

    # set up optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))

    # set up saved model's name
    save_file = save+'_'+args.loss_fn+'_'+str(args.layer)+'_'+str(args.n_epochs) + '_' + str(n_mat) + '_' + str(args.lambda_1) + '_'  + str(args.lambda_2) + '_' + str(args.batch_sz) + '_' + args.train.split('/')[-1]
    if args.batch_norm:
        save_file += '_bn'
    if args.skip_connection:
        save_file += '_sc'
    if args.pre_activation:
        save_file += '_pa'
    if args.patch_sz is not None:
        save_file += '_' + str(args.patch_sz) 
        save_file += '_' + str(args.n_patches)
    if args.rotate:
        save_file += '_rot'
    if args.hflip:
        save_file += '_flip'
    if args.standardize:
        save_file += '_std'
    if args.standardize_alt:
        save_file += '_stdalt'
        
    start_epoch = 0   
    running_loss = torch.zeros((args.n_epochs, 2))
    if args.resume_from is not None:
        pre_trained = torch.load('./results/checkpoints/'+args.resume_from+'.pt',map_location=torch.device('cpu'))
        start_epoch = pre_trained['epoch']+1
        model.load_state_dict(pre_trained['model'])
        optimizer.load_state_dict(pre_trained['optimizer'])
        #running_loss = pre_trained['loss']
        running_loss[0:start_epoch,:] = pre_trained['loss'][0:start_epoch,:] 
    
    if args.transfer_from is not None:
        model.load_state_dict(torch.load('./results/'+args.transfer_from+'.pt'))
        save_file += '_trf'
        
    # main loop 
    for epoch in range(start_epoch, args.n_epochs):
        print('Epoch: {}  \n ---------------------------'.format(epoch+1))
        running_loss[epoch,0] = train_loop(train_data, model, args.loss_fn, optimizer, args.lambda_1, args.lambda_2, vgg, vgg_alt, args.batch_sz, args.patch_sz, args.n_patches,device,args.display_interval,args.rotate,args.hflip)
        clear_output()
        running_loss[epoch,1] = test_loop(val_data , model, args.loss_fn, args.lambda_1, args.lambda_2, vgg, vgg_alt, args.n_samples, args.fig_sz, args.batch_sz, args.patch_sz, args.n_patches,device,args.rotate,args.hflip,args.standardize,args.standardize_alt,min_val,max_val,ct_trans)
        model_state = {'epoch': epoch, 
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': running_loss
            }
       # plt.plot(
      #      range(1,epoch+2),
      #      running_loss[0:(epoch+1),0],
      #      label="Train"
      #  )
      #  plt.plot(
      #      range(1,epoch+2),
      #      running_loss[0:(epoch+1),1],
      #      label="Val"
      #      )
      #  plt.title("Loss")
      #  plt.legend()
      #  plt.show()
        torch.save(model_state, './results/checkpoints/temp.pt')
        if epoch % args.log_interval == 0 and epoch!=0:
            torch.save(model_state, './results/checkpoints/'+save_file+'_'+str(epoch)+'.pt') 
             
    torch.save(model.state_dict(), './results/' + save_file + '.pt')
    torch.save(running_loss, './results/plots/' + save_file + '_plot.pt')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        )
    parser.add_argument(
        '--train',
        type = str,
        default = './data/train_rings_single_invmm_5_0.6_1.1_4',
        help = 'string indicating training set to be used',
        )
    parser.add_argument(
        '--valid',
        type = str,
        default = './data/val_rings_single_invmm_5_0.6_1.1_4',
        help = 'string indicating validation set to be used',
        )
    parser.add_argument(
        '--loss_fn',
        type = str,
        default = 'vgg16_l1',
        help = 'string indicating loss function to be used',
        )
    parser.add_argument(
        '--batch_sz',
        type = int,
        default = 2,
        help = 'batch size.')
    parser.add_argument(
        '--patch_sz',
        type = int,
        default = None,
        help = 'patch size.')
    parser.add_argument(
        '--n_patches',
        type = int,
        default = 1,
        help = 'number of patches extracted')
    parser.add_argument(
        '--device',
        type = int,
        default = 0,
        help = 'GPU to be used')
    parser.add_argument(
        '--layer',
        type = int,
        default = 9,
        help = 'layer used in vgg as feature extractor (Kim et al. use 23/24 in vgg16 and Yang et al. use 36 in vgg19)')
    parser.add_argument(
        '--n_iter',
        type = int,
        default = 10,
        help = 'number of iterations used in ResNet')
    parser.add_argument(
        '--n_channels',
        type = int,
        default = 32,
        help = 'number of channels used in ResNet')
    parser.add_argument(
        '--init_features',
        type = int,
        default = 64, 
        help = 'number of initial features used in Unet')
    parser.add_argument(
        '--num_resblocks',
        type = int, 
        default = 1, 
        help = '')
    parser.add_argument(
        '--ch_mult',
        type = str, 
        default = 'celebahq256', 
        help = '')
    parser.add_argument(
        '--n_epochs',
        type = int, 
        default = 350,
        help = 'number of epochs')
    parser.add_argument(
        '--net',
        type = str,
        default = 'sr3_alt_2VMI',
        help = 'string indicating network to be used (supported resnet/unet)',
        )
    parser.add_argument(
        '--batch_norm',
        dest = 'batch_norm',
        action='store_true',
        help = 'boolean indicating whether batch norm should be used in UNet'
        )
    parser.add_argument(
        '--no-batch_norm',
        dest = 'batch_norm',
        action='store_false',
        help = 'boolean indicating whether batch norm should be used in UNet'
        )
    parser.set_defaults(batch_norm=False)
    parser.add_argument(
        '--skip_connection',
        dest = 'skip_connection',
        action='store_true',
        help = 'boolean indicating whether skip connection should be used in UNet'
        )
    parser.add_argument(
        '--no-skip_connection',
        dest = 'skip_connection',
        action='store_false',
        help = 'boolean indicating whether skip_connection should be used in UNet'
        )
    parser.set_defaults(skip_connection=False)
    parser.add_argument(
        '--pre_activation',
        dest = 'pre_activation',
        action='store_true',
        help = 'boolean indicating whether pre-activation should be used in UNet'
        )
    parser.add_argument(
        '--no-pre_activation',
        dest = 'pre_activation',
        action='store_false',
        help = 'boolean indicating whether pre-activation should be used in UNet'
        )
    parser.set_defaults(pre_activation=False)
    parser.add_argument(
        '--rotate',
        dest = 'rotate',
        action='store_true',
        help = 'boolean indicating data augmentation via rotations should be used'
        )
    parser.add_argument(
        '--no-rotate',
        dest = 'rotate',
        action='store_false',
        help = 'boolean indicating data augmentation via rotations should be used'
        )
    parser.set_defaults(rotate=False)
    parser.add_argument(
        '--hflip',
        dest = 'hflip',
        action='store_true',
        help = 'boolean indicating data augmentation via rotations should be used'
        )
    parser.add_argument(
        '--no-hflip',
        dest = 'hflip',
        action='store_false',
        help = 'boolean indicating data augmentation via rotations should be used'
        )
    parser.set_defaults(hflip=False)
    parser.add_argument(
        '--standardize',
        dest = 'standardize',
        action='store_true',
        help = 'boolean indicating whether data should be mapped to [0,1]'
        )
    parser.add_argument(
        '--no-standardize',
        dest = 'standardize',
        action='store_false',
        help = 'boolean indicating whether data should be mapped to [0,1]'
        )
    parser.set_defaults(standardize=False)
    parser.add_argument(
        '--standardize_alt',
        dest = 'standardize_alt',
        action='store_true',
        help = 'boolean indicating whether data should be mapped to [0,1] using average min/max'
        )
    parser.add_argument(
        '--no-standardize_alt',
        dest = 'standardize_alt',
        action='store_false',
        help = 'boolean indicating whether data should be mapped to [0,1] using average min/max'
        )
    parser.set_defaults(standardize_alt=False)
    parser.add_argument(
        '--resume_from',
        type = str,
        default = None,
        help = 'resume training from checkpoint.')
    parser.add_argument(
        '--transfer_from',
        type = str,
        default = None,
        help = 'transfer training from result.')
    parser.add_argument(
        '--learning_rate',
        type = float,
        default = 1e-4,
        help = 'learning rate',
        )
    parser.add_argument(
        '--b1',
        type = float,
        default = 0.5,
        help = 'b1 parameter for ADAM',
        )
    parser.add_argument(
        '--b2',
        type = float,
        default = 0.9,
        help = 'b2 parameter for ADAM',
        )
    parser.add_argument(
        '--lambda_1',
        type = float,
        default = 1,
        help = 'Weight given to first loss objective',
        )
    parser.add_argument(
        '--lambda_2',
        type = float,
        default = 1,
        help = 'Weigh given to second loss objective',
        )
    parser.add_argument(
        '--log_interval',
        type = int,
        default = 5,
        help = '',
        )
    parser.add_argument(
        '--display_interval',
        type = int,
        default = 100,
        help = '',
        )
    parser.add_argument(
        '--n_samples',
        type = int,
        default = 2,
        help = '',
        )
    parser.add_argument(
        '--fig_sz',
        type = int,
        default = 10,
        help = '',
        )
    args = parser.parse_args()
    main(args)
