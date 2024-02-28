# Import packages
import argparse 
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import scipy.io as io
import os 

# user defined
import models 
import utils 
#import sr3
import sr3_alt
import sr3_alt_2VMI

def test(dataloader, model, loss_fn, n_samples, lambda_1, lambda_2, net, val_set, vgg, standardize, standardize_alt, save_as_pt):
    #n_mat = next(iter(dataloader)).size(1)//2
    n_mat = next(iter(dataloader)).size(1) - 1
    size = len(dataloader.dataset)
    
    # set up different loss configurations 
    loss_mse = nn.MSELoss() 
    loss_l1 = nn.L1Loss()
    
    loss_list = [] 
    psnr_list = []
    psnr_list_id = []
    
    ww = 70
    wl = 35

    model.eval()
        
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if standardize:
                data, val_min, val_max = utils.map_to_zero_one(data, return_vals=True)
            if standardize_alt:
                minmax = torch.load('./data/train_rings_cont_invmm_alt_alt_minmax.pt')
                data = utils.map_to_zero_one_alt(data, minmax[0], minmax[1])
                
            observed = data[:,0:n_mat,:,:] 
            truth = data[:,n_mat:n_mat*2,:,:] 
            
            if torch.cuda.is_available():
                observed = observed.cuda()
                truth = truth.cuda()
                model = model.cuda()
                if standardize:
                    val_min = val_min.cuda()
                    val_max = val_max.cuda()
                if standardize_alt:
                    minmax = minmax.cuda()
        
            # compute prediction and loss
            pred = model(observed)
            
            if loss_fn == 'mse':
                loss = loss_mse(pred, truth)
            elif loss_fn == 'mse_l1':
                loss = lambda_1*loss_mse(pred, truth) + lambda_2*loss_l1(pred,truth)
            elif loss_fn == 'l1':
                loss = loss_l1(pred, truth)
            elif loss_fn == 'vgg16' or loss_fn == 'vgg19':
                loss = loss_mse(vgg(pred), vgg(truth))
            elif loss_fn == 'vgg16_alt':
                loss = loss_mse(vgg(utils.get_mono(pred)),vgg(utils.get_mono(truth)))
            elif loss_fn == 'vgg16_mse' or loss_fn == 'vgg19_mse':
                loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_mse(pred,truth)
            elif loss_fn == 'vgg16_l1' or loss_fn == 'vgg19_l1':
                loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_l1(pred,truth)
            elif loss_fn == 'vgg16_l1_alt' or loss_fn =='vgg19_l1_alt':
                loss = lambda_1*loss_mse(vgg(utils.get_mono(pred)),vgg(utils.get_mono(truth)))+lambda_2*loss_l1(pred,truth)
            else:
                raise RuntimeError('Please provide a supported loss function')
                        
            # other performance metrics 
            #psnr = 10 * np.log10((torch.max(truth).item()**2) / loss_mse(pred,truth).item()) 
            #psnr_id = 10 * np.log10((torch.max(truth).item()**2) / loss_mse(observed,truth).item()) 
            psnr = 0
            psnr_id = 0

            # add to sum / list     
            loss_list += [loss.item()]
            psnr_list += [psnr]
            psnr_list_id += [psnr_id]

            if standardize:
                pred = utils.inv_map_to_zero_one(pred,val_min,val_max,'pred')
                observed = utils.inv_map_to_zero_one(observed,val_min,val_max,'obs')
                truth = utils.inv_map_to_zero_one(truth,val_min,val_max,'truth')
            if standardize_alt:
                pred = utils.inv_map_to_zero_one_alt(pred,minmax[0], minmax[1])
                observed = utils.inv_map_to_zero_one_alt(observed,minmax[0], minmax[1])
                truth = utils.inv_map_to_zero_one_alt(truth,minmax[0], minmax[1])
                
            
            if save_as_pt:
                save_dir_pt = './results/imgs_pt/'+net+'_' + val_set.split('/')[-1]+'/'
                try:
                    os.mkdir(save_dir_pt)
                except FileExistsError:
                    pass
                out = torch.cat((torch.cat((pred.detach().cpu(),observed.detach().cpu()),1), truth.detach().cpu()),1)
                torch.save(out,save_dir_pt+str(idx)+'.pt')
                
            save_dir = './results/imgs/'+net+'_' + val_set.split('/')[-1]+'/'
            try:
                os.mkdir(save_dir)
            except FileExistsError:
                pass 
             
            if idx <= n_samples:
                if idx < 3:
                    fig_sz = 10
                    #utils.plot_insert(truth[0,0,:,:].cpu(), cmap='gray', clim = [0, 2])
                    #utils.plot_insert(observed[0,0,:,:].cpu(), cmap= 'bone', clim = [0, 70])
                    #utils.plot_insert(pred[0,0,:,:].cpu(), cmap = 'gray', clim = [0, 2])
                    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True,
                                            figsize=(2 * fig_sz, fig_sz * 3))
                    axs[0, 0].imshow(observed[0,0,:,:].cpu(), cmap= 'bone', clim = [0, 70])
                    axs[1, 0].imshow(observed[0, 1, :, :].cpu(), cmap='bone', clim=[0, 70])
                    axs[0, 1].imshow(pred[0,0,:,:].cpu(), cmap = 'gray', clim = [0, 2])
                    axs[0, 2].imshow(truth[0,0,:,:].cpu(), cmap='gray', clim = [0, 2])

                    plt.show()
                    #if n_mat == 1:
                        #utils.plot_insert(utils.transform_CT(truth[0,0,:,:].cpu()),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                        #utils.plot_insert(utils.transform_CT(observed[0,0,:,:].cpu()),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                        #utils.plot_insert(utils.transform_CT(pred[0,0,:,:].cpu()),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                        #utils.plot_insert(utils.transform_CT(pred[0,0,:,:].cpu())-utils.transform_CT(truth[0,0,:,:].cpu()),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                    #else:
                        #virtual_truth_70 =  truth[:,0:1,:,:]*mu_soft_70 + truth[:,1:,:,:]*mu_bone_70
                        #virtual_obs_70 = observed[:,0:1,:,:]*mu_soft_70 + observed[:,1:,:,:]*mu_bone_70
                        #virtual_pred_70 = pred[:,0:1,:,:]*mu_soft_70 + pred[:,1:,:,:]*mu_bone_70

                        #utils.plot_insert(utils.transform_CT(virtual_truth_70[0,0,:,:].cpu()),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                        #utils.plot_insert(utils.transform_CT(virtual_obs_70[0,0,:,:].cpu()),cmap = 'bone',clim = [wl-ww/2,wl+ww/2])
                        #utils.plot_insert(utils.transform_CT(virtual_pred_70[0,0,:,:].cpu()),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                        #utils.plot_insert(utils.transform_CT(virtual_pred_70[0,0,:,:].cpu())-utils.transform_CT(virtual_truth_70[0,0,:,:].cpu()),cmap='bone',clim = [(wl-ww/2)*1e-2,(wl+ww/2)*1e-2])

                pred_save = pred.cpu()
                obs_save = observed.cpu()
                truth_save = truth.cpu()

                # Permute to fit order in matlab routine 
                #pred_save = pred_save.permute(0,3,2,1)
                #obs_save = obs_save.permute(0,3,2,1)
                #truth_save = truth_save.permute(0,3,2,1)

                # Move to numpy 
                pred_save = pred_save.numpy()
                obs_save = obs_save.numpy()
                truth_save = truth_save.numpy()

                # set up dictionaries 
                mat_dict = {'prediction' : pred_save}
                mat_dict.update({'observed' : obs_save})
                mat_dict.update({'truth' : truth_save})

                # save as .mat   
                io.savemat(save_dir+str(idx)+'_data.mat',mat_dict)
            
    print('Test results: basis images \n ---------------------------')
    print('Mean loss: {:.4f} '.format(np.mean(loss_list)))
    print('Std loss: {:.4f} '.format(np.std(loss_list)))
    print('Mean PSNR: {:.4f} '.format(np.mean(psnr_list)))
    print('Std PSNR: {:.4f} '.format(np.std(psnr_list)))
    print('Mean PSNR (id): {:.4f} '.format(np.mean(psnr_list_id)))
    print('Std PSNR (id): {:.4f} '.format(np.std(psnr_list_id)))
  
def main(args):
    # raise error if no model is given
    if args.net is None:
        raise RuntimeError('Please provide a model')
    
    # load data 
    data = torch.load(args.data + '.pt')
    n_mat = data.size(1) -1
    #img_sz = data.size(2)
    
    # set up trained model 
    string_split = args.net.split('_')
    batch_norm = False
    skip_connection = False
    pre_activation = False
    standardize = False
    standardize_alt = False 
    if 'bn' in string_split:
        batch_norm = True
    if 'sc' in string_split:
        skip_connection = True
    if 'pa' in string_split:
        pre_activation = True
    if 'std' in string_split:
        standardize = True
    if 'stdalt' in string_split:
        standardize_alt = True
    if 'celebahq' in string_split:
        ch_mult = (1, 2, 4, 8, 16, 32, 32, 32)
    elif 'celeba256' in string_split:
        ch_mult = (1, 1, 2, 2, 2, 2, 2)
    else:
        ch_mult = (1, 1, 2, 2, 2, 2, 2)    
    if string_split[0] == 'unet':
        if string_split[1] == 'alt':
            model = models.UNet_alt(n_mat, int(string_split[2]),norm=batch_norm,skip=skip_connection, pre=pre_activation) 
        else:
            model = models.UNet(n_mat, int(string_split[1]),norm=batch_norm) 
    elif string_split[0] == 'sr3':
        if string_split[1] == 'alt':
            model = sr3_alt_2VMI.NCSNpp(num_channels=n_mat, output_channels=1, nf=int(string_split[3]), num_resblocks=int(string_split[4]), image_size=args.patch_sz, skip=skip_connection, ch_mult=ch_mult)
        else:
            model = sr3.NCSNpp(num_channels=n_mat, nf=int(string_split[1]), num_resblocks=int(string_split[2]), image_size=args.patch_sz, skip=skip_connection)
    elif string_split[0] == 'yang':
        model = models.Generator_yang(n_mat, 32)
    elif string_split[0] == 'resnet':
        model = models.iterative_ResNet(int(string_split[1]), n_mat, int(string_split[2]))
    elif string_split[0] == 'cycle':
        if string_split[1] == 'alt':
            model = models.Generator_cycle_alt(n_mat, n_mat, int(string_split[2]))
        else:
            model = models.Generator_cycle(n_mat, n_mat, int(string_split[1]))
            
    else:
        raise RuntimeError('Please provide a supported model')
    
    # set up perceptual loss 
    vgg = None 
    if args.loss_fn=='vgg16_l1_alt' or args.loss_fn =='vgg16_alt': 
        vgg = models.VGG_Feature_Extractor_16(layer=args.layer, n_mat=1,requires_grad=False)
    elif args.loss_fn=='vgg19_l1_alt': 
        vgg = models.VGG_Feature_Extractor_16(layer=args.layer, n_mat=1,requires_grad=False)
    elif args.loss_fn.split('_')[0] == 'vgg16':
        vgg = models.VGG_Feature_Extractor_16(layer=args.layer,n_mat=1,requires_grad=False)
    elif args.loss_fn.split('_')[0] == 'vgg19':
        vgg = models.VGG_Feature_Extractor_19(layer=args.layer,n_mat=1,requires_grad=False)

    # move to cuda 
    if args.loss_fn.split('_')[0] == 'vgg16' or args.loss_fn.split('_')[0] == 'vgg19':
        vgg = vgg.cuda()
     
    model.load_state_dict(torch.load('./results/' + args.net + '.pt',map_location='cpu'))
        
    # set up dataloader 
    #data = (data - torch.min(data))/(torch.max(data)-torch.min(data)) 
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_sz, shuffle=False)
    
    if args.n_samples is None:
        n_samples = data.size(0)
    else:
        n_samples = args.n_samples

    # get performance metrics 
    test(dataloader, model, args.loss_fn, n_samples, args.lambda_1, args.lambda_2, args.net, args.data, vgg, standardize, standardize_alt, args.save_as_pt)
    
# for booleans see: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        )
    parser.add_argument(
        '--data',
        type = str,
        default = './data/nsclc_test',
        help = 'string indicating dataset to be used',
        )
    parser.add_argument(
        '--batch_sz',
        type = int,
        default = 1,
        help = 'batch size'
        )
    parser.add_argument(
        '--loss_fn',
        type = str,
        default = 'vgg16_l1',
        help = 'string indicating loss function to be used',
        )
    parser.add_argument(
        '--layer',
        type = int,
        default = 9,
        help = 'layer used in vgg as feature extractor (Kim et al. use 23/24 in vgg16 and Yang et al. use 36 in vgg19)')
    parser.add_argument(
        '--patch_sz',
        type = int,
        default = 256,
        help = 'patch size used in training (only relevant for NCSN++)')
    parser.add_argument(
        '--net',
        type = str,
        default = 'sr3_alt_2VMI_64_1_celebahq256new_for_check14_vgg16_l1_9_350_2_1_1_2_train_ICRU_pepvc_corr',
        help = 'string indicating network to be used (default: None)',
        )
    parser.add_argument(
        '--n_samples',
        type = int,
        default = 12,
        help = 'number of samples to be displayed. Note that this is only for plotting. The performance metrics use the entire dataset.'
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
        '--save_as_pt',
        dest = 'save_as_pt',
        action='store_true',
        help = ''
        )
    parser.add_argument(
        '--no-save_as_pt',
        dest = 'save_as_pt',
        action='store_false',
        help = ''
        )
    parser.set_defaults(save_as_pt=False)
    args = parser.parse_args()
    main(args)
    
