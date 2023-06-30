import os, path_n_util
if __name__ == '__main__':
    parser = path_n_util.main_train_parser()
    args = parser.parse_args()
    os.environ['CUDA VISIBLE DEVICES'] = args.CUDA

import scanpy as sc
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
# from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from functools import partial
from src import _sampler, _epsilon_module, _helper_net, _learner, _reader, _configure


#                   _                          
##                 | \  _. _|_  _.             ##
##                 |_/ (_|  |_ (_|             ##

def dl_from_config(configs, shuffle=True, n_workers=4):
    adata = sc.read(configs.anndata_path)


    # - DataSet - 
    configs.check_samples = False
    ds_fn = partial(eval(f"_reader.{configs.dataset_class}"), adata,  **configs.dataset_kwargs)
    
    AnDatasets = [ds_fn(which_set=s) for s in ['train','val','test']]

    # - DataLoader - 
    def my_collate(batch):
        "Puts each data field into a tensor with outer dimension batch size"
        if len(batch[0]) == 8:
            batch = list( filter (lambda x: x[5] - x[1] == configs.collect_time_span, batch))
        # if np.any([not isinstance(x, torch.Tensor) for x in batch[0]]):
        #     new_batch = []
        #     for item in batch:
        #         new_batch.append([torch.Tensor(x) for x in item if not isinstance(x, torch.Tensor)])
        #     batch = new_batch
        return default_collate(batch)

    train_loader = DataLoader(AnDatasets[0], batch_size = configs.batch_size, 
                                collate_fn = my_collate,
                                pin_memory = False,
                                shuffle = shuffle, num_workers=n_workers)
    val_loader = DataLoader(AnDatasets[1], batch_size = configs.batch_size, 
                                collate_fn = my_collate,
                                pin_memory = False,
                                shuffle = False, num_workers=n_workers)
    test_loader = DataLoader(AnDatasets[2], batch_size = configs.batch_size, 
                                collate_fn = my_collate,
                                pin_memory = False,
                                shuffle = False, num_workers=n_workers)
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

    device = torch.device("cuda:%s"%args.CUDA) if torch.cuda.is_available() else 'cpu'
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    config_dir = os.path.basename(os.path.dirname(args.model_config))
    configs = _configure.Yaml_configurer(args.model_config)
    configs.check_samples = False

    train_loader, val_loader, test_loader = dl_from_config(configs, n_workers=args.n_workers)
    eps_net = get_model_from_config(configs, args.CUDA)
    Sampler = get_sampler_from_configs(configs, eps_net)


    run_name = os.path.basename(args.model_config).replace(".yaml","")
    log_dir = os.path.join(path_n_util.pth_dir , config_dir, "{}_{}".format(configs.epsilon_class, run_name))

    trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1,
            auto_lr_find=True,
            # fast_dev_run = True,
            overfit_batches = args.overfit_batches,
            default_root_dir=log_dir,
            max_epochs=200, 
            auto_select_gpus = False,
            check_val_every_n_epoch=1,
            callbacks=[
                callbacks.ModelCheckpoint(save_top_k=1, monitor="val_loss"),
                callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=50)
                ])        

    trainer.fit(Sampler, train_loader, val_loader)
    
    trainer.validate(Sampler, test_loader)
    trainer.test(Sampler, test_loader)
