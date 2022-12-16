#!/usr/bin/env python
# coding: utf-8
# +
from fastai.distributed import *
from fastai.vision.all import *
import torch
import numpy as np
import os
from glob import glob

print("fold 1 validation - folds 2,3,4,5 training")


# -

def label_func(fn):
    path_mask = Path("/data/weed_data/kfold_5ds/labels")
    return path_mask/f"{fn.parent.stem}"/ f"{fn.stem.replace('image', 'mask')}{fn.suffix}"

# +
# Not used but maybe can be useful in the future
# def custom_splitter(f, val_fold="1"):
#     "divide between train/val datasets based on the fold"
#     def _inner(f, val_fold):
#         val_idx = mask2idxs([i.parent.stem == val_fold for i in f])
#         train_idx = np.setdiff1d(np.array(range_of(f)), np.array(val_idx))
#         return L(train_idx, use_list=True), L(val_idx, use_list=True)
#     return _inner
# -

if __name__ == "__main__":
    results = {"val_loss":[],
          "dice":[],
          "iou":[]}

    # Prepare datasets
    # Load the data
    codes = np.loadtxt("/data/weed_data/inputImages_Masks/codes.txt", dtype='str')
    path = Path('/data/weed_data/kfold_5ds/images')
    
    #define the valid idx
    a = get_image_files(path, folders=("1","2", "3", "4", "5"))
    valid_idx = mask2idxs([i.parent.stem == "1" for i in a])
      
    # load the data
    weedt = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                  get_items = get_image_files,
                  get_y = label_func,
                  splitter=IndexSplitter(valid_idx),
                  batch_tfms=Normalize)      
    
    # create the data loader
    dls= weedt.dataloaders(path, folders=("1","2", "3", "4", "5"), path=path, bs=4)

    # build the model
    learn = unet_learner(dls,
                     resnet18,
                     n_out=3,
                     self_attention=True,
                     normalize=True,
                     pretrained=True,
                     loss_func=DiceLoss(),
                     opt_func=Adam,
                     lr= 0.001,
                     wd=None,
                     metrics=[DiceMulti, JaccardCoeff]).to_fp16()

    # Disable Fastai progress bar
    with learn.no_bar() and learn.distrib_ctx():
         learn.fit_one_cycle(2, 1e-5)
         learn.unfreeze()
         learn.fit_one_cycle(100,
                            lr_max= slice(1e-7, 1e-5),
                            cbs=CSVLogger(fname='/data/weed_data/kfold_5ds/unet18_kfold_1VAL', append=True))

    learn.save('/data/weed_data/model_weights/unet18_1fold', 
                 with_opt=True,
                 pickle_protocol=2)

    learn.export(fname='/data/weed_data/model_weights/unet18_1fold_export.pkl',
                   pickle_module=pickle,
                   pickle_protocol=2)

