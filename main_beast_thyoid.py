from lib import *
from utils import get_args_parser, cal_contrastiveloss
from ema import ExponentialMovingAverage
from Dataset import split_dataset, multi_dataset
from engine_train import train_one_epoch, train_one_epoch_nextphrase, train_one_epoch_linprobe_Inc, train_one_epoch_linprobe
from engine_val import evaluate_old_new, inference
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(args):
    misc.init_distributed_mode(args)
    torch.autograd.detect_anomaly = True
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = args.device == 'cuda'
    # dataset
    train_task_datasets, val_task_datasets, train_all_loaders, Inc_loader, val_loaders, train_loaders, test_split_loaders= multi_dataset(args, args.data_path_list, args.n_tasks, train_batch_size=args.batch_size, test_batch_size=args.batch_size, Inc_img_nums=args.Inc_img_nums)
    print(train_task_datasets)

    if args.distributed:  # args.distributed
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_task_datasets, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_task_datasets)

    if args.distributed and global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    # define the model
    model_before = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                                   lambda_ratio=args.lambda_ratio).float()

    model_after = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                                  lambda_ratio=args.lambda_ratio).float()
    forget_matrix = [[0.697, 0.0]]
    # forget_matrix = []
    start_time = time.time()
    for task_idx in range(1, len(train_all_loaders)):
        print(f"Start training for {task_idx} Task")
        current_train_all_loader = train_all_loaders[task_idx]
        current_train_loader = train_loaders[task_idx]
        current_val_loader = val_loaders[task_idx]
        current_Inc_loader = Inc_loader[task_idx]
        #beast_thyoid_DAT_Inc
        output_dir = f'/archive/zcz/beast_thyoid_DAT_Inc_{args.Inc_img_nums}/Inc_task{task_idx}/'
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        finetune_model_path = f'/archive/zcz/beast_thyoid_DAT_Inc_{args.Inc_img_nums}/Inc_task{task_idx-1}/checkpoint-best.pth'
        finetune_linprobe_model_path = f'/archive/zcz/beast_thyoid_DAT_Inc_{args.Inc_img_nums}/Inc_task{task_idx-1}_linprobe/checkpoint-best.pth'
        if task_idx != 0:
            checkpoint = torch.load(finetune_model_path, map_location='cpu')['model']
            checkpoint_linprobe = torch.load(finetune_linprobe_model_path, map_location='cpu')['model']
            print("Load pre-trained checkpoint from: %s" % finetune_model_path)
            # for key in checkpoint:
            #     if key in checkpoint_linprobe:  # 如果键存在于model_after的state_dict中
            #         # 用old_model_state_dict中对应键的值替换model_after中的值
            #         checkpoint[key] = checkpoint_linprobe[key]

            # load pre-trained model
            msg0 = model_before.load_state_dict(checkpoint, strict=False)
            msg1 = model_after.load_state_dict(checkpoint, strict=False)
            print(msg0)
            print(msg1)
        if task_idx == 0:
            checkpoint = torch.load(args.finetune, map_location='cpu')['model']
            msg0 = model_before.load_state_dict(checkpoint, strict=True)
            msg1 = model_after.load_state_dict(checkpoint, strict=True)
            print(msg0)
            print(msg1)

        model_before.to(device)
        model_after.to(device)
        for _, p in model_before.named_parameters():
            p.requires_grad = True
        for _, p in model_after.named_parameters():
            p.requires_grad = True

        model_without_ddp = model_after
        print("Model = %s" % str(model_without_ddp))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)
        if args.distributed and global_rank == 0 and args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None
        if args.distributed:
            model_before = torch.nn.parallel.DistributedDataParallel(model_before, device_ids=[args.gpu],
                                                                     find_unused_parameters=True)
            model_after = torch.nn.parallel.DistributedDataParallel(model_after, device_ids=[args.gpu],
                                                                    find_unused_parameters=True)
            # model = torch.nn.DataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model_after.module

        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer_pretrain = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        loss_scaler_mae = NativeScaler()
        loss_scaler_mask = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer_pretrain, loss_scaler=loss_scaler_mae)

        optimizer_Incphrase = torch.optim.AdamW(model_after.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("############### Inc phrase ##############")
        if task_idx != 0:
            ############## Inc phrase ###########
            min_loss = float('inf')
            for epoch in range(args.start_epoch, args.epochs_pretrain_Inc):
                if args.distributed:
                    current_Inc_loader.sampler.set_epoch(epoch)

                if epoch < (args.epochs_pretrain / 2):
                    curriculum_factor = 1 - (2 * epoch) / (args.epochs_pretrain_Inc)
                else:
                    curriculum_factor = 1 - (2 * (epoch + 1)) / (args.epochs_pretrain_Inc)
                # current_Inc_loader
                train_stats = train_one_epoch_nextphrase(
                    model_before, model_after, current_Inc_loader,
                    optimizer_Incphrase, device, epoch, curriculum_factor, loss_scaler_mae, loss_scaler_mask,
                    log_writer=log_writer,
                    args=args
                )
                log_stats1 = {**{f'train_Inc_{k}': v for k, v in train_stats.items()},
                              'epoch': epoch, }
                current_loss = log_stats1['train_Inc_loss']  # 请替换 'your_loss_key' 为实际的损失值键名
                if current_loss < min_loss:
                    min_loss = current_loss
                    # 保存具有最小损失的模型
                    misc.save_model(
                        args=args, model=model_after, model_without_ddp=model_after, optimizer=optimizer_Incphrase,
                        loss_scaler=loss_scaler_mae, epoch='best', output_dir=output_dir)

                if output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(output_dir, "log_masknet.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats1) + "\n")

        print("############### pretrain phrase ##############")
        print(f"Start training for {args.epochs_pretrain} epochs")
        min_loss = float('inf')
        for epoch in range(args.start_epoch, args.epochs_pretrain):
            if args.distributed:
                current_train_loader.sampler.set_epoch(epoch)

            if epoch < (args.epochs_pretrain / 2):
                curriculum_factor = 1 - (2 * epoch) / (args.epochs_pretrain)
            else:
                curriculum_factor = 1 - (2 * (epoch + 1)) / (args.epochs_pretrain)

            current_loss = train_one_epoch(
                model_after, current_train_loader,
                optimizer_pretrain, device, epoch, curriculum_factor, loss_scaler_mae, loss_scaler_mask,
                log_writer=log_writer,
                args=args
            )
            if current_loss < min_loss:
                min_loss = current_loss
                min_loss_epoch = epoch
                # 保存具有最小损失的模型
                misc.save_model(
                    args=args, model=model_after, model_without_ddp=model_after, optimizer=optimizer_pretrain,
                    loss_scaler=loss_scaler_mae, epoch='best', output_dir=output_dir)

            if output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    # f.write(json.dumps(log_stats) + "\n")
                    f.write(f"Epoch: {epoch}, Loss: {current_loss}, Min Loss: {min_loss} (Epoch: {min_loss_epoch})\n")

        print('############### linprobe phrase ###############')
        nb_classes = args.nb_classes_list[-1]
        model_linprobe = models_vit.__dict__['vit_base_patch16'](
            num_classes=nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        model_linprobe.to(device)
        model_linprobe_old = models_vit.__dict__['vit_base_patch16'](
            num_classes=nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        model_linprobe_old.to(device)
        output_dir_linprobe = f'/archive/zcz/beast_thyoid_DAT_Inc_{args.Inc_img_nums}/Inc_task{task_idx}_linprobe/'
        if output_dir_linprobe:
            Path(output_dir_linprobe).mkdir(parents=True, exist_ok=True)
        if task_idx != 0:
            finetune_dir_linprobe = f'/archive/zcz/beast_thyoid_DAT_Inc_{args.Inc_img_nums}/Inc_task{task_idx - 1}_linprobe/'
            old_model_state_dict = torch.load(finetune_dir_linprobe + '/checkpoint-best.pth')['model']
            msg = model_linprobe_old.load_state_dict(old_model_state_dict, strict=True)
            print("Load old model_linprobe state_dict success ->" + str(msg))
            new_model_state_dict = old_model_state_dict.copy()
            # 遍历old_model_state_dict的键
            for key in new_model_state_dict:
                if key in model_after.state_dict():  # 如果键存在于model_after的state_dict中
                    # 用model_after中对应键的值替换old_model_state_dict中的值
                    new_model_state_dict[key] = model_after.state_dict()[key]

            for _, p in model_linprobe.named_parameters():
                p.requires_grad = True

            old_out_features = args.nb_classes_list[task_idx - 1],  # 旧模型的输出特征数量

            msg = model_linprobe.load_state_dict(new_model_state_dict, strict=True)
            print("Load model_linprobe state_dict success ->" + str(msg))

            for _, p in model_linprobe.named_parameters():
                if 'head' in _.split('.'):
                    p.requires_grad = True  # 仅将分类头的参数设置为需要梯度
                else:
                    p.requires_grad = True  # 主干(backbone)的参数设置为不需要梯度


        else:
            new_model_state_dict = model_linprobe.state_dict()
            for key in new_model_state_dict:
                if key in model_after.state_dict():  # 如果键存在于model_after的state_dict中
                    # 用model_after中对应键的值替换old_model_state_dict中的值
                    new_model_state_dict[key] = model_after.state_dict()[key]

            # load pre-trained model
            msg = model_linprobe.load_state_dict(new_model_state_dict, strict=False)
            print(msg)
            # manually initialize fc layer
            timm.models.layers.trunc_normal_(model_linprobe.head.weight, std=2e-5)
            for _, p in model_linprobe.named_parameters():
                if 'head' in _.split('.'):
                    p.requires_grad = True  # 仅将分类头的参数设置为需要梯度
                else:
                    p.requires_grad = True  # 主干(backbone)的参数设置为不需要梯度
            model_linprobe.to(device)


        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        # optimizer_linprobe = torch.optim.SGD(
        #     model_linprobe.parameters(),
        #     lr=1e-3,
        #     momentum=0.9,
        #     weight_decay=0.0005,
        # )  # 1e-5
        # # [60, 120, 170]
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer=optimizer_linprobe, milestones= [30, 60, 90], gamma= 0.1
        # )

        param_groups_linprobe = lrd.param_groups_lrd(model_linprobe, args.weight_decay,
                                            no_weight_decay_list=model_linprobe.no_weight_decay(),
                                            layer_decay=args.layer_decay
                                            )
        optimizer_linprobe = torch.optim.AdamW(param_groups_linprobe, lr=args.lr)

        loss_scaler = NativeScaler()
        n_parameters = sum(p.numel() for p in model_linprobe.parameters() if p.requires_grad)
        misc.load_model(args=args, model_without_ddp=model_linprobe, optimizer=optimizer_linprobe, loss_scaler=loss_scaler)
        print(f"Start linprobe training for {args.finetune_epochs} epochs")
        max_accuracies = [0] * (len(train_all_loaders))
        max_accuracy = 0
        expert_max_accuracy = 0
        max_accuracies_templist = [0] * (len(train_all_loaders))
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        if task_idx != 0:
            # Example usage
            ema = ExponentialMovingAverage(beta=0.9)
            ema.register(model_linprobe_old)

        for epoch in range(args.start_epoch, args.finetune_epochs):
            if args.distributed:
                current_train_all_loader.sampler.set_epoch(epoch)
            if task_idx!=0:
                for _, p in model_linprobe_old.named_parameters():
                    p.requires_grad = False
                train_stats = train_one_epoch_linprobe_Inc(
                    model_linprobe, model_linprobe_old, criterion, current_Inc_loader,
                    optimizer_linprobe, device, epoch, loss_scaler,
                    args.clip_grad, None,
                    log_writer=log_writer,
                    args=args, old_out_features=old_out_features
                )
                train_stats = train_one_epoch_linprobe_Inc(
                    model_linprobe, model_linprobe_old, criterion, current_train_all_loader,
                    optimizer_linprobe, device, epoch, loss_scaler,
                    args.clip_grad, None,
                    log_writer=log_writer,
                    args=args, old_out_features=old_out_features
                )
                # if epoch > args.finetune_epochs*2:
                # # if epoch > -1:
                #     ema.update(model_linprobe)  # Update EMA after each epoch

            else:
                train_stats = train_one_epoch_linprobe(
                    model_linprobe, criterion, current_train_all_loader,
                    optimizer_linprobe, device, epoch, loss_scaler,
                    args.clip_grad, None,
                    log_writer=log_writer,
                    args=args
                )
            # if task_idx!=0 and epoch > args.finetune_epochs / 2:
            # if task_idx!=0 and epoch > -1:
            #     ema.update(model_linprobe)  # Update EMA after each epoch
            #     ema.apply_shadow(model_linprobe_old)
            # scheduler.step()
            ema.apply_shadow(model_linprobe_old)
            print('新模型在所有任务测试')
            test_stats = evaluate_old_new(current_val_loader, model_linprobe, device)
            print("新模型在旧任务evaluate_old:")
            test_stats_old = evaluate_old_new(test_split_loaders[0], model_linprobe, device)
            print("新模型在新任务evaluate_new:")
            test_stats_new = evaluate_old_new(test_split_loaders[1], model_linprobe, device)
            if task_idx == 0:
                if test_stats["acc1"] > max_accuracy:
                    misc.save_model(
                        args=args, model=model_linprobe, model_without_ddp=model_linprobe,
                        optimizer=optimizer_linprobe,
                        loss_scaler=loss_scaler, epoch='best', output_dir=output_dir_linprobe)
                    print(f"Best accuracy saved pth achieved at epoch {epoch}.")
                    max_accuracy = test_stats["acc1"]
                max_accuracy_msg = f'Max accuracy: {max_accuracy * 100:.2f}%'
                print(max_accuracy_msg)

            elif task_idx!= 0:
                print("ema更新后的model_linprobe_old在旧任务:")
                ema_old = evaluate_old_new(test_split_loaders[0], model_linprobe_old, device)
                print("ema更新后的model_linprobe_old在新任务:")
                ema_new = evaluate_old_new(test_split_loaders[1], model_linprobe_old, device)
                print("专家在所有任务测试:")
                expert_all = inference(current_val_loader, model_linprobe, model_linprobe_old, device)
                print(f"Accuracy of the network on the {len(current_val_loader.dataset)} test images: {expert_all['acc1'] * 100:.1f}%")

                if expert_all["acc1"] > max_accuracy:
                    misc.save_model(
                        args=args, model=model_linprobe, model_without_ddp=model_linprobe,
                        optimizer=model_linprobe,
                        loss_scaler=loss_scaler, epoch='best', output_dir=output_dir_linprobe)
                    print(f"Best accuracy saved pth achieved at epoch {epoch}.")
                    max_accuracy = expert_all["acc1"]
                max_accuracy_msg = f'Max accuracy: {max_accuracy * 100:.2f}%'
                print(max_accuracy_msg)

            print("新模型/专家在各个任务测试:")
            experts_results = {}

            for i in range(len(train_all_loaders)):
                print('task_{}:'.format(i), end='')
                test_stats_i = evaluate_old_new(test_split_loaders[i], model_linprobe, device)
                if task_idx != 0:
                    expert = inference(test_split_loaders[i], model_linprobe, model_linprobe_old, device,
                                       threshold=[args.nb_classes_list[i-1],args.nb_classes_list[i]])
                    experts_results[f'expert_{i}_acc1'] = expert["acc1"]
                    # experts_results[f'expert_{i}_acc5'] = expert.get("acc5", None)  # Assuming
                    # if expert["acc1"] > max_accuracies[i]:
                    max_accuracies[i] = expert["acc1"]
                else:
                    # if test_stats_i["acc1"] > max_accuracies[i]:
                    max_accuracies[i] = test_stats_i["acc1"]
            if expert_max_accuracy < sum(max_accuracies)/len(train_all_loaders):
                expert_max_accuracy = sum(max_accuracies)/len(train_all_loaders)
                print('expert_max_accuracy', expert_max_accuracy)
                max_accuracies_templist = max_accuracies.copy()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         **experts_results,
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         'max_accuracy_msg': max_accuracy_msg}
            if output_dir_linprobe and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(output_dir_linprobe, "log_linprobe_0.9.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        if args.calforget:
            forget_matrix.append(max_accuracies_templist)
    if args.calforget:
        task = len(train_all_loaders)-1
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(forget_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (CNN):')
        print('forgetting: ', forgetting)
        print('np_acctable: ', np_acctable)
        # 创建一个包含这些值的字典
        forget_log_stats = {
            'forgetting': forgetting,
            'np_acctable': np_acctable.T.tolist()
        }
        if output_dir_linprobe and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir_linprobe, "log_linprobe_0.9.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(forget_log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)