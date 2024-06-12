import argparse
import torch

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--n_tasks', default=2, type=int, help='task number')
    # parser.add_argument('--n_tasks', default=3, type=int, help='task number')
    parser.add_argument('--calforget', action='store_true',
                        help='if or not calculate forgetting')
    parser.set_defaults(calforget=True)
    parser.add_argument('--Inc_img_nums', default=25, type=int, help='Increment number')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs_pretrain', default=1, type=int)
    parser.add_argument('--epochs_pretrain_Inc', default=2, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--finetune_epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    parser.add_argument('--nb_classes_list', default=[3,6], type=int, help='number of the classification types')
    # parser.add_argument('--nb_classes_list', default=[25,33,39], type=int, help='number of the classification types')
    # Masking V2 parameters
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--lambda_ratio', default=2.0, type=float)
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--finetune', default='/archive/zcz1/dafenlei_Inc_500_DAT_noFree/dafenlei_Inc_task0/checkpoint-best.pth', help='finetune from checkpoint')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path_list',
                        default=['/archive/zhuchunzheng/dafenlei/reorganize_breast_thyroid/AUITD_Thyroid_Dataset200/', '/archive/zhuchunzheng/dafenlei/reorganize_breast_thyroid/BUSI_nomask200/'], type=str, help='dataset path')
    # parser.add_argument('--data_path_list',
    #                     default=['/archive/zhuchunzheng/dafenlei/midlate_500_resize/',
    #                              '/archive/zhuchunzheng/dafenlei/early_data_500_resize/',
    #                              '/archive/zhuchunzheng/dafenlei/fuke_data_500_resize/'], type=str, help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    # distributed training parameters
    parser.add_argument('--gpu', nargs='+', type=str, help='gpus')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def cal_contrastiveloss( img1_rep, img2_rep, bidirect_contrast=True):
    # calculate nce loss for mean-visual representation and mean-audio representation

    img1_rep = torch.nn.functional.normalize(img1_rep, dim=-1)
    img2_rep = torch.nn.functional.normalize(img2_rep, dim=-1)
    img1_rep = img1_rep.view(img1_rep.size()[0], -1)
    img2_rep = img2_rep.view(img2_rep.size()[0], -1)
    total = torch.mm(img1_rep, torch.transpose(img2_rep, 0, 1)) / 0.05

    # by default we use single directional
    if bidirect_contrast == False:
        nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
        c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0),
                                   torch.arange(0, total.shape[0], device=img1_rep.device))) / total.shape[0]
        return nce, c_acc
    else:
        nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
        nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
        c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0),
                                     torch.arange(0, total.shape[0], device=img1_rep.device))) / total.shape[0]
        c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0),
                                     torch.arange(0, total.shape[0], device=img1_rep.device))) / total.shape[0]
        nce = (nce_1 + nce_2) / 2
        c_acc = (c_acc_1 + c_acc_2) / 2
        return nce, c_acc



