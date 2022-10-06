from distutils.command.config import config
import os, path_n_util
parser = path_n_util.main_train_parser()
args = parser.parse_args()
os.environ['CUDA VISIBLE DEVICES'] = args.CUDA
import logging
import torch
from torch import nn
from torch import optim
import numpy as np
import scanpy as sc
from torchsummary import summary
from torch.utils.data import DataLoader
from functools import partial
from src import _sampler, _epsilon_module, _helper_net, _sampler, _reader, _configure

configs = _configure.Json_configurer(args.model_config)
               

#                   _                          
##                 | \  _. _|_  _.             ##
##                 |_/ (_|  |_ (_|             ##

adata = sc.read(os.path.join(path_n_util.data_dir, configs.anndata_path))
# TODO: unique_token_dict fun from config
adata.obs['pert1'] = adata.obs['condition'].apply(lambda x: x.split("+")[0])
adata.obs['pert2'] = adata.obs['condition'].apply(lambda x: x.split("+")[-1])
# unique tokens
unique_token = list(set(adata.obs['pert1'].tolist() + adata.obs['pert2'].tolist()))
unique_token.remove('ctrl')
unique_token = ['ctrl'] + unique_token
unique_token_dict = {token:i for i, token in enumerate(unique_token)}

# - DataSet - 
ds_fn = partial(_reader.Condition_AnnDataSet, adata, unique_token_dict, **configs.dataset_kwargs())
AnDatasets = (ds_fn(which_set=s) for s in ['train','test','ood'])
# - DataLoader - 
dl_fn = partial(DataLoader, batch_size=configs.batch_size, shuffle=True)
train_loader, val_loader, test_loader = (dl_fn(Dataset=ds) for ds in AnDatasets)


##             |\/|  _   _|     |  _           ##
##             |  | (_) (_| |_| | (/_          ##

# - device -
if torch.cuda.is_available():
    device = torch.device("cuda:%d"%args.CUDA)
else:
    device = torch.device('cpu')

module_kw = configs.module_kwargs()
Module_Class = eval("_epsilon_module.%s" %configs.module_class)

# - pretrain - 
if configs.pretrain_embedder_pth is not None:
    pretrain_embedder = torch.load(configs.pretrain_embedder_pth)
    if configs.fix_pretrain:    
        # fix embeddings at the beginning
        for p in pretrain_embedder.parameters():
            p.required_grads = False
    module_kw['pretrained_embeddings'] = pretrain_embedder

# - define module -
eps_net = Module_Class(**module_kw).to(device)

# - optimizer -
optimizer = optim.Adam(eps_net.parameters())



#              __                                #
#             (_   _. ._ _  ._  |  _  ._         #
#             __) (_| | | | |_) | (/_ |          #
#                           |                    #

Samper_Class = eval("_sampler.%s" %configs.sampler_class)
sampler_kwargs = configs.sampler_kwargs()
sampler_kwargs['model'] = eps_net

Sampler = Samper_Class(**module_kw)

#
#
logging.basicConfig(filename=args.model_config.replace(".yaml", '.log'), level=logging.INFO)

best_loss = 1e8
for epoch in range(configs.epochs):
    #train
    eps_net.train()
    logging.info("=======    {epoch %d}    =======" %epoch)
    for step, minibatch in enumerate(train_loader):
        optimizer.zero_grad()
        
        if configs.use_batch_index:
            X, exp_batch, c = minibatch
            exp_batch = exp_batch.to(device)
            
        else:
            X, c = minibatch
            exp_batch = None
        
        X = X.float().to(device)
        c = c.to(device)
        
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        T_b = torch.randint(0, configs.timesteps, (configs.batch_size, ), device=device).long()
        
        loss = Sampler.p_losses(eps_net,  T_b, loss_type='huber')
        
        if step %100 == 0:
            logging.info("\tLoss:%.6f" % loss.item())
        
        loss.backward()
        optimizer.step()
    
    # val
    eps_net.eval()
    with torch.no_grad():
        all_loss = []
        for step, minibatch in enumerate(val_loader):

            if configs.use_batch_index:
                X, exp_batch, c = minibatch
                exp_batch = exp_batch.to(device)
                
            else:
                X, c = minibatch
                exp_batch = None
            
            X = X.float().to(device)
            c = c.to(device)
            
            L = []
            for t in range(0, configs.timesteps, 20):
                T_b = torch.full((configs.batch_size,), t, device=device).long()
                L.append(Sampler.p_losses(eps_net,  T_b, loss_type='huber'))
            loss = np.mean(L.cpu().numpy())
            all_loss.append(loss)
            logging.info("=======    {val %d}    =======" %epoch)
            logging.info("\tLoss:%.6f" % np.mean(all_loss).item())

        if best_loss > np.mean(all_loss):
            all_loss = all_loss

            model_pth = os.path.join(path_n_util.pth_dir , configs.model_class, args.model_config.replace(".yaml",".pth"))
            torch.save(eps_net, model_pth)