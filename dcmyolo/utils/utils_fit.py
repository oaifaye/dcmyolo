import os

import torch
from tqdm import tqdm

from dcmyolo.utils.utils_data import get_lr
        
def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, cur_epoch, epoch_step, epoch_step_val, gen, gen_val, epoch, cuda, fp16, scaler, save_period, save_dir):
    loss        = 0
    val_loss    = 0

    print('Start Train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {cur_epoch + 1}/{epoch}', postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets, y_trues = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda()
                targets = [ann.cuda() for ann in targets]
                y_trues = [ann.cuda() for ann in y_trues]
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs         = model_train(images)

            loss_value_all  = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all += loss_item
            loss_value = loss_value_all

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs         = model_train(images)

                loss_value_all  = 0
                # ----------------------#
                #   计算损失
                # ----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                    loss_value_all += loss_item
                loss_value = loss_value_all

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
                            'lr'    : get_lr(optimizer)})
        pbar.update(1)

    pbar.close()
    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {cur_epoch + 1}/{epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, y_trues = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda()
                targets = [ann.cuda() for ann in targets]
                y_trues = [ann.cuda() for ann in y_trues]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train_eval(images)

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all  += loss_item
            loss_value  = loss_value_all

        val_loss += loss_value.item()
        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
        pbar.update(1)
            
    pbar.close()
    print('Finish Validation')
    loss_history.append_loss(cur_epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    eval_callback.on_epoch_end(cur_epoch + 1, model_train_eval)
    print('Epoch:' + str(cur_epoch + 1) + '/' + str(epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    
    #-----------------------------------------------#
    #   保存权值
    #-----------------------------------------------#
    if ema:
        save_state_dict = ema.ema.state_dict()
    else:
        save_state_dict = model.state_dict()

    if (cur_epoch + 1) % save_period == 0 or cur_epoch + 1 == epoch:
        torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (cur_epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
        
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
    print('Save best model to ', os.path.join(save_dir, "last_epoch_weights.pth"))
    torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))