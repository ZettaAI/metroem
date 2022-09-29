import torch
import time
import torchfields

from torch.optim.lr_scheduler import ReduceLROnPlateau

from metroem import helpers
from metroem.loss import unsupervised_loss


def combine_pre_post(res, post):
    result = post.from_pixels()(res.from_pixels()).pixels()
    return result

def optimize_pre_post_ups(src, tgt, initial_res, sm, lr, num_iter,
                      src_defects,
                      tgt_defects,
                      src_zeros,
                      tgt_zeros,
                      opt_params=None,
                      opt_mode='adam',
                      crop=16,
                      noimpr_period=50,
                      opt_res_coarsness=0,
                      wd=0,
                      l2=1e-4,
                      normalize=True,
                      sm_keys_to_apply=None,
                      mse_keys_to_apply=None,
                      verbose=False,
                      max_bad=15
                    ):
    if opt_params is None:
        opt_params = {}
    if sm_keys_to_apply is None:
        sm_keys_to_apply = {}
    if mse_keys_to_apply is None:
        mse_keys_to_apply = {}

    opti_loss = unsupervised_loss(
          smoothness_factor=sm, use_defect_mask=True,
          sm_keys_to_apply=sm_keys_to_apply,
          mse_keys_to_apply=mse_keys_to_apply
      )
    pred_res = initial_res.detach().field()
    if opt_res_coarsness > 0:
        pred_res = pred_res.down(opt_res_coarsness)
    pred_res.requires_grad = True

    trainable = [pred_res]
    if opt_mode == 'adam':
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    elif opt_mode == 'sgd':
        optimizer = torch.optim.SGD(trainable, lr=lr, **opt_params)

    min_lr = lr * (0.5**max_bad)

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.5,
        patience=noimpr_period,
        verbose=True,
        threshold_mode='rel',
        threshold=1e-5,
        cooldown=25,
        min_lr=min_lr,
    )

    if normalize:
        with torch.no_grad():
            src_mask = torch.logical_not(src_zeros)
            tgt_mask = torch.logical_not(tgt_zeros)

            while src_mask.ndim < src.ndim:
                src_mask.unsqueeze_(0)
            while tgt_mask.ndim < src.ndim:
                tgt_mask.unsqueeze_(0)

            src = helpers.normalize(src, mask=src_mask, mask_fill=0)
            tgt = helpers.normalize(tgt, mask=tgt_mask, mask_fill=0)
    loss_bundle = {
        'src': src,
        'tgt': tgt,
        'src_defects': src_defects,
        'tgt_defects': tgt_defects,
        'src_zeros': src_zeros,
        'tgt_zeros': tgt_zeros,
    }

    s = time.time()

    loss_bundle['pred_res'] = pred_res
    if opt_res_coarsness > 0:
        loss_bundle['pred_res'] = loss_bundle['pred_res'].up(opt_res_coarsness)
    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
    loss_dict = opti_loss(loss_bundle, crop=crop)
    if verbose:
        print (loss_dict['result'].detach().cpu().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())

    for epoch in range(num_iter):
        loss_bundle['pred_res'] = pred_res
        if opt_res_coarsness > 0:
            loss_bundle['pred_res'] = loss_bundle['pred_res'].up(opt_res_coarsness)
        loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
        loss_dict = opti_loss(loss_bundle, crop=crop)
        loss_var = loss_dict['result']
        if l2 > 0.0:
            loss_var += (loss_bundle['pred_res']**2).mean() * l2

        optimizer.zero_grad()
        loss_var.backward()

        torch.nn.utils.clip_grad_norm_(trainable, 0.49)  # attempt to prevent flipped edges
        scheduler.step(loss_var)
        if optimizer.param_groups[0]['lr'] <= min_lr:
            break
        optimizer.step()

    loss_bundle['pred_res'] = pred_res
    if opt_res_coarsness > 0:
        loss_bundle['pred_res'] = loss_bundle['pred_res'].up(opt_res_coarsness)
    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
    loss_dict = opti_loss(loss_bundle, crop=crop)

    e = time.time()

    if verbose:
        print ("Iter: {}".format(epoch))
        print (loss_dict['result'].detach().cpu().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        print (e - s)
        print ('==========')

    return loss_bundle['pred_res']
