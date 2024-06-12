from lib import *

def evaluate_old_new(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    num_samples = len(data_loader.dataset)
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, target = batch[0].to(device), batch[-1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # calculate accuracy
            _, pred_top1 = output.topk(1, dim=1)
            _, pred_top5 = output.topk(3, dim=1)
            correct_top1 += pred_top1.eq(target.view(-1, 1)).sum().item()
            correct_top5 += pred_top5.eq(target.view(-1, 1)).sum().clamp(max=1).item()

            total_loss += loss.item() * images.size(0)

            if (i + 1) % 10 == 0:
                print(f'Test: [{i + 1}/{len(data_loader)}]')

    avg_loss = total_loss / num_samples
    avg_acc1 = correct_top1 / num_samples
    avg_acc5 = correct_top5 / num_samples
    avg_acc1 = round(avg_acc1, 3)
    avg_acc5 = round(avg_acc5, 3)
    avg_loss = round(avg_loss, 3)
    print(f'* Acc@1 {avg_acc1:.3f} Acc@5 {avg_acc5:.3f} loss {avg_loss:.3f}')

    return {'acc1': avg_acc1, 'acc5': avg_acc5, 'loss': avg_loss}

def inference(data_loader, model, model_old, device, threshold=None):
    criterion = torch.nn.CrossEntropyLoss()

    num_samples = len(data_loader.dataset)
    total_loss_ens = 0.0
    correct_top1_ens = 0
    correct_top5_ens = 0

    pred_on, pred_off, pred_ens, gts = [], [], [], []

    # Switch to evaluation mode
    model.eval()
    model_old.eval()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, target = batch[0].to(device), batch[1].to(device)

            # Compute outputs for both models
            output_new = model(images)
            output_old = model_old(images)

            # Compute loss
            # loss = criterion(output_new, target)

            # Integrate outputs
            output_new_soft = torch.softmax(output_new, dim=1)
            output_old_soft = torch.softmax(output_old, dim=1)

            # output_ens = torch.stack([output_new_soft, output_old_soft], dim=-1).max(dim=-1)[0]
            # output_ens = torch.stack([output_new, output_old], dim=-1).max(dim=-1)[0]

            # threshold = 3  # 示例阈值，用于区分新旧任务
            # is_new_task = target >= threshold
            # 定义阈值范围
            # threshold = [a, b]
            # 判断 target 是否在范围 [a, b) 内
            if threshold!=None:
                is_new_task = (target >= threshold[0]) & (target < threshold[1])

                alpha = 0.8  # 新任务输出的权重
                beta = 0.2  # 旧任务输出的权重

                output_ens = torch.where(is_new_task.unsqueeze(1),
                                         alpha * output_new_soft + beta * output_old_soft,
                                         beta * output_new_soft + alpha * output_old_soft)
            else:
                output_ens = torch.stack([output_new_soft, output_old_soft], dim=-1).max(dim=-1)[0]
            # output_new_soft_ens = torch.where(is_new_task.unsqueeze(1), alpha * output_new_soft,  beta * output_new_soft)
            # output_old_soft_ens = torch.where(is_new_task.unsqueeze(1), beta * output_old_soft,  alpha * output_old_soft)
            # output_ens = torch.stack([output_new_soft_ens, output_old_soft_ens], dim=-1).max(dim=-1)[0]
            # Calculate ensemble loss
            loss_ens = criterion(output_ens, target)

            # Calculate accuracy for ensemble predictions
            _, pred_top1_ens = output_ens.topk(1, dim=1)
            _, pred_top5_ens = output_ens.topk(3, dim=1)
            correct_top1_ens += pred_top1_ens.eq(target.view(-1, 1)).sum().item()
            correct_top5_ens += pred_top5_ens.eq(target.view(-1, 1)).sum().clamp(max=1).item()

            total_loss_ens += loss_ens.item() * images.size(0)

            # Save predictions and ground truths
            pred_on.append(output_new.argmax(dim=1))
            pred_off.append(output_old.argmax(dim=1))
            pred_ens.append(output_ens.argmax(dim=1))
            gts.append(target)

            if (i + 1) % 10 == 0:
                print(f'Test: [{i + 1}/{len(data_loader)}]')

    # Calculate averaged metrics for ensemble predictions
    avg_loss_ens = total_loss_ens / num_samples
    avg_acc1_ens = correct_top1_ens / num_samples
    avg_acc5_ens = correct_top5_ens / num_samples

    avg_acc1_ens = round(avg_acc1_ens, 3)
    avg_acc5_ens = round(avg_acc5_ens, 3)
    avg_loss_ens = round(avg_loss_ens, 3)

    # print(f'* Ensemble Acc@1 {avg_acc1_ens:.3f} Acc@5 {avg_acc5_ens:.3f} loss {avg_loss_ens:.3f}')
    print(f'* Ensemble Acc@1 {avg_acc1_ens:.3f} loss {avg_loss_ens:.3f}')

    return {'acc1': avg_acc1_ens, 'acc5': avg_acc5_ens, 'loss': avg_loss_ens}

