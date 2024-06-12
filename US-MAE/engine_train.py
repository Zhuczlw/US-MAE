from lib import *
from utils import cal_contrastiveloss

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, curriculum_factor: float, loss_scaler, loss_scaler_mask,
                    log_writer=None,
                    args=None):
    model.train(True)
    model = model.float()
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    #################### Initiating Sequence: Alternative Training Masking-Net and MAE ######################################################################
    total_loss = 0.0
    num_batches = len(data_loader)

    for data_iter_step, (samples, _) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_batches + epoch, args)

        samples = samples.to(device, non_blocking=True)

        ######################################### Train MAE ######################################3
        model.freeze_maskingnet()
        with torch.cuda.amp.autocast():
            loss_recon, _ = model(samples, mask_ratio=args.mask_ratio)

        loss = loss_recon
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        model.unfreeze_maskingnet()

        ######################################### Train Masking-Net ######################################
        model.freeze_backbone()
        with torch.cuda.amp.autocast():
            loss_recon, fixedRatio_loss, loss_diversity = model(samples, mask_ratio=args.mask_ratio,
                                                                    train_mask=True)

        # print("lambda_cl : {} ".format(curriculum_factor))
        loss = curriculum_factor * loss_recon + fixedRatio_loss + loss_diversity
        loss_value_mm = loss.item()

        if not math.isfinite(loss_value_mm):
            sys.exit(1)

        loss /= accum_iter
        loss_scaler_mask(loss, optimizer, parameters=model.parameters(),
                         update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        model.unfreeze_backbone()

        total_loss += loss_value

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / num_batches + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch_1000x)

        if (data_iter_step + 1) % print_freq == 0:
            print(f"Epoch: [{epoch}]  Step: [{data_iter_step+1}/{num_batches}]  "
                  f"Loss: {loss_value:.6f}  LR: {optimizer.param_groups[0]['lr']:.6f}")

    average_loss = total_loss / num_batches
    print(f"Epoch: [{epoch}]  Average Loss: {average_loss:.6f}")

    return average_loss

def train_one_epoch_nextphrase(model_before: torch.nn.Module,
                    model_after: torch.nn.Module,
                    data_loader_Inc: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, curriculum_factor: float, loss_scaler, loss_scaler_mask,
                    log_writer=None,
                    args=None):
    model_before.train(False)
    model_after.train(True)
    model_before = model_before.float()
    model_after = model_after.float()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    model_after.masking_net.load_state_dict(model_before.masking_net.state_dict())

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
        #############Inc#############
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader_Inc, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_Inc) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        ######################################### Train MAE ######################################3
        model_after.freeze_maskingnet()
        model_before.freeze_maskingnet()
        model_before.freeze_backbone()

        with torch.cuda.amp.autocast():
            loss_recon, latent_after = model_after(samples, mask_ratio=args.mask_ratio)
            _, latent_before = model_before(samples, mask_ratio=args.mask_ratio)
        contrasloss, _ = cal_contrastiveloss(latent_after, latent_before)  ## A+B
        loss = loss_recon + contrasloss
        # print(loss_recon , contrasloss) ####
        # loss = loss_recon
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model_after.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        model_after.unfreeze_maskingnet()

        ######################################### Train Masking-Net ######################################
        model_after.freeze_backbone()
        with torch.cuda.amp.autocast():
            loss_recon, fixedRatio_loss, loss_diversity = model_after(samples, mask_ratio=args.mask_ratio,
                                                                          train_mask=True)
        print( loss_recon, fixedRatio_loss, loss_diversity)
        # print("lambda_cl : {} ".format(curriculum_factor))
        loss = curriculum_factor * loss_recon + fixedRatio_loss + loss_diversity
        loss_value_mm = loss.item()

        if not math.isfinite(loss_value_mm):
            sys.exit(1)

        loss /= accum_iter
        loss_scaler_mask(loss, optimizer, parameters=model_after.parameters(),
                         update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        model_after.unfreeze_backbone()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader_Inc) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_linprobe_Inc(model: torch.nn.Module, model_old: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, old_out_features=0, alpha = 0.8, T = 2):
    model.train(True)
    model_old.train(False)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 2
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs_teacher = model_old(samples)
            loss1 = criterion(outputs, targets)

            outputs_S = torch.nn.functional.softmax(outputs / T, dim=1)
            outputs_T = torch.nn.functional.softmax(outputs_teacher / T, dim=1)
            loss2 = outputs_T.mul(-1 * torch.log(outputs_S))
            loss2 = loss2.sum(1)
            distillation_loss = loss2.mean() * T * T


            loss = loss1 * alpha + distillation_loss * (1 - alpha)
            # print(loss1, distillation_loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_linprobe(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}