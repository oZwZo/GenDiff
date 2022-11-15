import os, path_n_util
if __name__ == '__main__':
    parser = path_n_util.main_train_parser()
    args = parser.parse_args()
    os.environ['CUDA VISIBLE DEVICES'] = args.CUDA

import scanpy as sc
import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from functools import partial
from src import _sampler, _epsilon_module, _helper_net, _learner, _reader, _configure



               

#                   _                          
##                 | \  _. _|_  _.             ##
##                 |_/ (_|  |_ (_|             ##

def dl_from_config(configs, shuffle=True):
    adata = sc.read(configs.anndata_path)
    # TODO: unique_token_dict fun from config
    # adata.obs['pert1'] = adata.obs['condition'].apply(lambda x: x.split("+")[0])
    # adata.obs['pert2'] = adata.obs['condition'].apply(lambda x: x.split("+")[-1])
    # # unique tokens
    # unique_token = list(set(adata.obs['pert1'].tolist() + adata.obs['pert2'].tolist()))
    # unique_token.remove('ctrl')
    # unique_token = ['ctrl'] + unique_token
    # unique_token_dict = {token:i for i, token in enumerate(unique_token)}

    # - DataSet - 
    ds_fn = partial(eval(f"_reader.{configs.dataset_class}"), adata,  **configs.dataset_kwargs)
    # AnDatasets = (ds_fn(which_set=s) for s in ['train','test','ood'])
    AnDatasets = [ds_fn(which_set=s) for s in ['train','val','test']]
    # - DataLoader - 
    def my_collate(batch):
        "Puts each data field into a tensor with outer dimension batch size"
        if len(batch[0]) == 8:
            batch = list( filter (lambda x: x[5] - x[1] == configs.collect_time_span, batch))
        return default_collate(batch)
    dl_fn = partial(DataLoader, batch_size=configs.batch_size, 
                                collate_fn = my_collate,
                                shuffle=False, num_workers=4)
    train_loader, val_loader, test_loader = (dl_fn(ds) for ds in AnDatasets)
    return train_loader, val_loader, test_loader


##             |\/|  _   _|     |  _           ##
##             |  | (_) (_| |_| | (/_          ##
def get_model_from_config(configs, cuda):

    # - device -
    if torch.cuda.is_available():
        device = torch.device("cuda:%s"%cuda)
    else:
        device = torch.device('cpu')

    module_kw = configs.epsilon_kwargs
    Module_Class = eval("_epsilon_module.%s" %configs.epsilon_class)

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
    return eps_net



#              __                                #
#             (_   _. ._ _  ._  |  _  ._         #
#             __) (_| | | | |_) | (/_ |          #
#                           |                    #

def get_sampler_from_configs(configs, eps_net):
    Samper_Class = eval("_learner.%s" %configs.sampler_class)
    sampler_kwargs = configs.sampler_kwargs
    sampler_kwargs['model'] = eps_net

    Sampler = Samper_Class(**sampler_kwargs)
    return Sampler



#              ___                              #
#               | ._ _. o ._   _  ._            #
#               | | (_| | | | (/_ |             #

if __name__ == '__main__':
    configs = _configure.Yaml_configurer(args.model_config)

    train_loader, val_loader, test_loader = dl_from_config(configs)
    eps_net = get_model_from_config(configs, args.CUDA)
    Sampler = get_sampler_from_configs(configs, eps_net)


    run_name = os.path.basename(args.model_config).replace(".yaml","")
    log_dir = os.path.join(path_n_util.pth_dir , configs.sampler_class, "{}_{}".format(configs.epsilon_class, run_name))

    trainer = pl.Trainer(
            accelerator='gpu', devices=1,
            auto_lr_find=True,
            default_root_dir=log_dir,
            max_epochs=configs.epochs, 
            auto_select_gpus = True,
            callbacks=[callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=15)])        

    trainer.fit(Sampler, train_loader, val_loader)
    trainer.validate(Sampler, test_loader)
    trainer.test(Sampler, test_loader)
