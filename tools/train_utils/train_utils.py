import glob
import os
from pathlib import Path

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
try:
    import wandb
except ImportError:
    wandb = None


def _is_torch_scheduler(scheduler):
    lrscheduler_cls = getattr(torch.optim.lr_scheduler, 'LRScheduler', None)
    if lrscheduler_cls is not None and isinstance(scheduler, lrscheduler_cls):
        return True
    return isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)


def _step_scheduler(scheduler, cur_iter):
    if scheduler is None:
        return
    if _is_torch_scheduler(scheduler):
        scheduler.step()
    else:
        scheduler.step(cur_iter)


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, use_wandb=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    scheduler_is_torch = _is_torch_scheduler(lr_scheduler) if lr_scheduler is not None else False

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        if lr_scheduler is not None and not scheduler_is_torch:
            lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        if scheduler_is_torch:
            _step_scheduler(lr_scheduler, accumulated_iter)

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

            if use_wandb and wandb is not None:
                wandb_dict = {'train/loss': loss, 'meta_data/learning_rate': cur_lr}
                for key, val in tb_dict.items():
                    wandb_dict['train/' + key] = val
                wandb.log(wandb_dict, step=accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_wandb=False,
                eval_loader=None, eval_model=None, eval_func=None, eval_output_dir=None,
                eval_interval=1, early_stop_cfg=None, dist_test=False):
    accumulated_iter = start_iter

    if rank == 0:
        Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    best_metric = None
    best_epoch = -1
    bad_epochs = 0

    def _get_cfg(cfg_dict, key, default=None):
        if cfg_dict is None:
            return default
        if key in cfg_dict:
            return cfg_dict.get(key, default)
        lower_key = key.lower()
        if lower_key in cfg_dict:
            return cfg_dict.get(lower_key, default)
        return default

    def _is_improved(cur_metric, best_val, mode, min_delta):
        if best_val is None:
            return True
        if mode == 'min':
            return cur_metric < best_val - min_delta
        return cur_metric > best_val + min_delta

    def _resolve_early_stop_metric(eval_ret, cfg_dict):
        metric_keys = _get_cfg(cfg_dict, 'METRICS', None)
        if metric_keys is None:
            metric_key = _get_cfg(cfg_dict, 'METRIC', None)
            if metric_key is None or metric_key not in eval_ret:
                return None, None
            return float(eval_ret[metric_key]), str(metric_key)

        if not isinstance(metric_keys, (list, tuple)) or len(metric_keys) == 0:
            return None, None
        if not all(key in eval_ret for key in metric_keys):
            return None, None

        metric_vals = [float(eval_ret[key]) for key in metric_keys]
        reducer = str(_get_cfg(cfg_dict, 'METRIC_REDUCER', 'mean')).lower()
        metric_weights = _get_cfg(cfg_dict, 'METRIC_WEIGHTS', None)
        if metric_weights is not None:
            if not isinstance(metric_weights, (list, tuple)) or len(metric_weights) != len(metric_vals):
                raise ValueError('METRIC_WEIGHTS must have same length as METRICS')
            metric_weights = [float(w) for w in metric_weights]

        if reducer in ['mean', 'avg']:
            score = sum(metric_vals) / len(metric_vals)
        elif reducer == 'sum':
            score = sum(metric_vals)
        elif reducer == 'min':
            score = min(metric_vals)
        elif reducer == 'max':
            score = max(metric_vals)
        elif reducer in ['weighted_mean', 'wmean']:
            if metric_weights is None:
                metric_weights = [1.0] * len(metric_vals)
            denom = sum(metric_weights)
            score = sum(v * w for v, w in zip(metric_vals, metric_weights)) / max(denom, 1e-12)
        else:
            raise ValueError('Unsupported METRIC_REDUCER: %s' % reducer)

        return float(score), '%s(%s)' % (reducer, ','.join(metric_keys))

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                use_wandb=use_wandb
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            if eval_loader is None or eval_func is None or eval_output_dir is None:
                continue

            if trained_epoch % max(eval_interval, 1) != 0:
                continue

            if early_stop_cfg is None:
                continue

            if not _get_cfg(early_stop_cfg, 'ENABLED', False):
                continue

            start_es_epoch = int(_get_cfg(early_stop_cfg, 'START_EPOCH', 0))
            if trained_epoch < start_es_epoch:
                continue

            eval_model_to_use = eval_model if eval_model is not None else model
            cur_result_dir = eval_output_dir / ('epoch_%s' % trained_epoch)
            eval_ret = eval_func(
                eval_model_to_use, eval_loader, trained_epoch, result_dir=cur_result_dir, dist_test=dist_test
            )
            model.train()
            if eval_model_to_use is not model:
                eval_model_to_use.train()

            if rank != 0:
                continue

            cur_metric, metric_name = _resolve_early_stop_metric(eval_ret, early_stop_cfg)
            if cur_metric is None:
                continue

            mode = str(_get_cfg(early_stop_cfg, 'MODE', 'max')).lower()
            min_delta = float(_get_cfg(early_stop_cfg, 'MIN_DELTA', 0.0))
            patience = int(_get_cfg(early_stop_cfg, 'PATIENCE', 10))

            if _is_improved(cur_metric, best_metric, mode, min_delta):
                best_metric = cur_metric
                best_epoch = trained_epoch
                bad_epochs = 0
                if _get_cfg(early_stop_cfg, 'SAVE_BEST', True):
                    ckpt_name = ckpt_save_dir / 'checkpoint_best'
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if rank == 0:
                        print(
                            'Early stopping at epoch %d (best %s=%.6f at epoch %d)'
                            % (trained_epoch, metric_name, best_metric, best_epoch)
                        )
                    break


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = Path('{}.pth'.format(filename))
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(filename))
