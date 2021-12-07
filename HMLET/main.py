import world
import dataloader
import Procedure
import utils

import pandas as pd
import math
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
from os.path import join

utils.set_seed(world.SEED)

dataset = dataloader.Loader(world.DATA_PATH)

model = world.MODELS[world.model_name](world.config, dataset)
model = model.to(world.device)
bpr = utils.BPRLoss(model)

# Pretrain
if world.pretrain:
    try:
        pretrained_file = world.LOAD_FILE_PATH
        model.load_state_dict(torch.load(pretrained_file))
        print(f"loaded model weights from {pretrained_file}")
    except FileNotFoundError:
        print(f"{pretrained_file} not exists, start from beginning")

# Tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-"))
                                    )
else:
    w = None
    print("not enable tensorflowboard")

try:
    start_epoch = world.load_epoch
    gum_temp = world.ori_temp
    for epoch in range(start_epoch, world.EPOCHS+1):
        start = time.time()
        
        print('Train', epoch, '='*30)
        print('gum_temp:', gum_temp)
        output_information = Procedure.BPR_train_original(dataset, model, bpr, world.EPOCHS, gum_temp, hard=world.train_hard, w=w)
        print(f'EPOCH[{epoch}/{world.EPOCHS}] {output_information}')
        
        end = time.time()
        print('train time:', end-start)
        
        if epoch % world.epoch_temp_decay == 0:
            # Temp decay
            gum_temp = world.ori_temp * math.exp(-world.gum_temp_decay*epoch)
            gum_temp = max(gum_temp, world.min_temp)
            print('decay gum_temp:', gum_temp)
        
        if epoch % 10 == 0:
            print("model save...")
            torch.save(model.state_dict(), world.SAVE_FILE_PATH+'/'+world.model_name+'_'+world.dataset+'_'+str(world.ori_temp)+'_'+str(world.gum_temp_decay)+'_'+str(world.min_temp)+'_'+str(world.epoch_temp_decay)+'_'+str(world.config['division_noise'])+'_'+str(epoch)+".pth.tar")
            
            print('Valid', '='*50)
            valid_results, valid_excel_data = Procedure.Test(dataset, model, epoch, gum_temp, hard=world.test_hard, mode='valid', w=w, multicore=world.multicore)

            xlxs_dir = world.EXCEL_PATH + '/valid_'+str(world.model_name)+'_'+str(world.dataset)+'_'+str(world.config['embedding_dim'])+'_'+str(world.ori_temp)+'_'+str(world.gum_temp_decay)+'_'+str(world.min_temp)+'_'+str(world.epoch_temp_decay)+'_'+str(world.config['division_noise'])+'_'+str(world.config['dropout'])+'_'+str(world.config['keep_prob'])+'_'+str(world.topks)+'.xlsx'
        
            with pd.ExcelWriter(xlxs_dir) as writer:
                valid_excel_data.to_excel(writer, sheet_name = 'result')            
            
            print('Test', '='*50)
            test_results, test_excel_data = Procedure.Test(dataset, model, epoch, gum_temp, hard=world.test_hard, mode='test', w=w, multicore=world.multicore)
            
            xlxs_dir = world.EXCEL_PATH + '/test_'+str(world.model_name)+'_'+str(world.dataset)+'_'+str(world.config['embedding_dim'])+'_'+str(world.ori_temp)+'_'+str(world.gum_temp_decay)+'_'+str(world.min_temp)+'_'+str(world.epoch_temp_decay)+'_'+str(world.config['division_noise'])+'_'+str(world.config['dropout'])+'_'+str(world.config['keep_prob'])+'_'+str(world.topks)+'.xlsx'
        
            with pd.ExcelWriter(xlxs_dir) as writer:
                test_excel_data.to_excel(writer, sheet_name = 'result')
            
finally:
    if world.tensorboard:
        w.close()