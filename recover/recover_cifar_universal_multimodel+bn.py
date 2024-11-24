import os
import random
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
from PIL import Image
from utils import BNFeatureHook, lr_cosine_policy, save_images, clip_image, denormalize_image
# import wandb


def clip_cifar(image_tensor):
    """
    adjust the input based on mean and variance for cifar
    """
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize_cifar(image_tensor):
    """
    convert floats back to input for cifar
    """
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def get_images(args, model_teacher_1, model_teacher_2, model_teacher_3, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size
    best_cost = 1e4
    
    # bn just from model_teacher_1
    loss_r_feature_layers_1 = []
    for module in model_teacher_1.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers_1.append(BNFeatureHook(module))
    loss_r_feature_layers_2 = []
    for module in model_teacher_2.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers_2.append(BNFeatureHook(module))
    loss_r_feature_layers_3 = []
    for module in model_teacher_3.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers_3.append(BNFeatureHook(module))
    # setup target labels
    targets_all = torch.LongTensor(np.arange(100))

    for kk in range(0, 100, batch_size):
        targets = targets_all[kk : min(kk + batch_size, 100)].to("cuda")

        data_type = torch.float
        # inputs = torch.randn((targets.shape[0], 3, 32, 32), requires_grad=True, device="cuda", dtype=data_type)
        # partly reinit_zx
        loaded_tensor = torch.load(args.init_path+'/tensor_'+str(ipc_id%25)+'.pt').clone().requires_grad_(True).to('cuda')
        input_original = loaded_tensor.detach().clone()
        input_original.requires_grad_(False)
        input_original = input_original.to('cuda')
        uni_perb = torch.randn((1, 3, 32, 32), requires_grad=True, device="cuda", dtype=data_type)
        # uni_perb_1 = torch.zeros((1, 3, 32, 32), requires_grad=True, device="cuda", dtype=data_type)
        # uni_perb_2 = torch.zeros((1, 3, 32, 32), requires_grad=True, device="cuda", dtype=data_type)
        # uni_perb_3 = torch.zeros((1, 3, 32, 32), requires_grad=True, device="cuda", dtype=data_type)
        uni_perb = uni_perb.to('cuda')
        # uni_perb_1 = uni_perb_1.to('cuda')
        # uni_perb_2 = uni_perb_2.to('cuda')
        # uni_perb_3 = uni_perb_3.to('cuda')

        # partly reinit_zx
        iterations_per_layer = args.iteration
        # ipc - 1 init plus 1 universal perb
        if ipc_id < args.init_part_num:#(args.ipc_end - 1):
            iterations_per_layer = 0
            inputs = input_original
        else:
            inputs = input_original + uni_perb

            # inputs_1 = input_original + uni_perb_1
            # inputs_2 = input_original + uni_perb_2
            # inputs_3 = input_original + uni_perb_3


        lim_0, lim_1 = args.jitter, args.jitter

        optimizer = optim.Adam([uni_perb], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)
            # apply random jitter offsets
            inputs = input_original + uni_perb
            # inputs_1 = input_original + uni_perb_1
            # inputs_2 = input_original + uni_perb_2
            # inputs_3 = input_original + uni_perb_3

            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

            # inputs_jit_1 = torch.roll(inputs_1, shifts=(off1, off2), dims=(2, 3))
            # inputs_jit_2 = torch.roll(inputs_2, shifts=(off1, off2), dims=(2, 3))
            # inputs_jit_3 = torch.roll(inputs_3, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            outputs_1 = model_teacher_1(inputs_jit)
            outputs_2 = model_teacher_2(inputs_jit)
            outputs_3 = model_teacher_3(inputs_jit)


            # R_cross classification loss
            loss_ce_1 = criterion(outputs_1, targets)
            loss_ce_2 = criterion(outputs_2, targets)
            loss_ce_3 = criterion(outputs_3, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers_1) - 1)]
            loss_r_bn_feature_1 = [
                mod.r_feature.to(loss_ce_1.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers_1)
            ]
            loss_r_bn_feature_1 = torch.stack(loss_r_bn_feature_1).sum()
            loss_aux_1 = args.r_bn * loss_r_bn_feature_1
            #####
            rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers_2) - 1)]
            loss_r_bn_feature_2 = [
                mod.r_feature.to(loss_ce_1.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers_2)
            ]
            loss_r_bn_feature_2 = torch.stack(loss_r_bn_feature_2).sum()
            loss_aux_2 = args.r_bn * loss_r_bn_feature_2
            #####
            rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers_3) - 1)]
            loss_r_bn_feature_3 = [
                mod.r_feature.to(loss_ce_1.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers_3)
            ]
            loss_r_bn_feature_3 = torch.stack(loss_r_bn_feature_3).sum()
            loss_aux_3 = args.r_bn * loss_r_bn_feature_3

            loss = 1/3*(loss_ce_1+ loss_ce_2 + loss_ce_3) + 1/3*(loss_aux_1 +loss_aux_2 +loss_aux_3)

            if iteration % save_every == 0 and args.verifier:
                print("------------iteration {}----------".format(iteration))
                print("loss_ce_1", loss_ce_1.item())
                print("loss_ce_2", loss_ce_2.item())
                print("loss_ce_3", loss_ce_3.item())

                print("loss_r_bn_feature", loss_r_bn_feature_1.item())
                print("loss_r_bn_feature", loss_r_bn_feature_2.item())
                print("loss_r_bn_feature", loss_r_bn_feature_3.item())

                print("loss_total", loss.item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    acc_jit, _ = hook_for_display(inputs_jit, targets)
                    acc_image, loss_image = hook_for_display(inputs, targets)

                #     metrics = {
                #         'crop/acc_crop': acc_jit,
                #         'image/acc_image': acc_image,
                #         'image/loss_image': loss_image,
                #     }
                #     wandb_metrics.update(metrics)

                # metrics = {
                #     'crop/loss_ce': loss_ce.item(),
                #     'crop/loss_r_bn_feature': loss_r_bn_feature.item(),
                #     'crop/loss_total': loss.item(),
                # }
                # wandb_metrics.update(metrics)
                # wandb.log(wandb_metrics)

            # do image update
            loss.backward()
            # __import__('pdb').set_trace()
            optimizer.step()
            # clip color outlayers
            # inputs.data = clip_image(inputs.data, "cifar")

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = denormalize_image(best_inputs, "cifar")
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

    torch.cuda.empty_cache()

def main_syn(args, ipc_id):
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    # model_teacher = models.__dict__[args.arch_name](pretrained=True)

    import torchvision
    # multi models
    ## model 1
    model_teacher_1 = torchvision.models.resnet18(num_classes=100)
    model_teacher_1.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model_teacher_1.maxpool = nn.Identity()
    model_teacher_1 = nn.DataParallel(model_teacher_1).cuda()
    checkpoint_1 = torch.load(args.arch_path_1)
    model_teacher_1.load_state_dict(checkpoint_1["state_dict"])
    model_teacher_1.eval()
    for p in model_teacher_1.parameters():
        p.requires_grad = False
    ## model 2
    model_teacher_2 = torchvision.models.resnet34(num_classes=100)
    model_teacher_2.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model_teacher_2.maxpool = nn.Identity()
    model_teacher_2 = nn.DataParallel(model_teacher_2).cuda()
    checkpoint_2 = torch.load(args.arch_path_2)
    model_teacher_2.load_state_dict(checkpoint_2["state_dict"])
    model_teacher_2.eval()
    for p in model_teacher_2.parameters():
        p.requires_grad = False
    ## model 3
    model_teacher_3 = torchvision.models.resnet50(num_classes=100)
    model_teacher_3.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model_teacher_3.maxpool = nn.Identity()
    model_teacher_3 = nn.DataParallel(model_teacher_3).cuda()
    checkpoint_3 = torch.load(args.arch_path_3)
    model_teacher_3.load_state_dict(checkpoint_3["state_dict"])
    model_teacher_3.eval()
    for p in model_teacher_3.parameters():
        p.requires_grad = False
  
    hook_for_display = None

    get_images(args, model_teacher_1, model_teacher_2, model_teacher_3, hook_for_display, ipc_id)


def parse_args():
    parser = argparse.ArgumentParser("SRe2L: recover data from pre-trained model")
    """Data save flags"""
    parser.add_argument(
        "--exp-name", type=str, default="recoverd_result", help="name of the experiment, subfolder under syn_data_path"
    )
    parser.add_argument("--syn-data-path", type=str, default="../syn_data", help="where to store synthetic data")
    parser.add_argument("--store-best-images", action="store_true", help="whether to store best images")
    """Optimization related flags"""
    parser.add_argument("--batch-size", type=int, default=100, help="number of images to optimize at the same time")
    parser.add_argument("--iteration", type=int, default=1000, help="num of iterations to optimize the synthetic data")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate for optimization")
    parser.add_argument("--jitter", default=4, type=int, help="random shift on the synthetic data")
    parser.add_argument(
        "--r-bn", type=float, default=0.05, help="coefficient for BN feature distribution regularization"
    )
    parser.add_argument(
        "--first-bn-multiplier", type=float, default=10.0, help="additional multiplier on first bn layer of R_bn"
    )
    """Model related flags"""
    parser.add_argument(
        "--arch-name", type=str, default="resnet18", help="arch name from pretrained torchvision models"
    )
    parser.add_argument("--arch-path-1", type=str, default="/home/dujw/Robustness/otho_SRE/codes/save/cifar100/resnet18_E200/ckpt.pth")
    parser.add_argument("--arch-path-2", type=str, default="/home/dujw/Robustness/otho_SRE/codes/save/cifar100/resnet34_E200/ckpt.pth")
    parser.add_argument("--arch-path-3", type=str, default="/home/dujw/Robustness/otho_SRE/codes/save/cifar100/resnet50_E200/ckpt.pth")

    parser.add_argument("--verifier", action="store_true", help="whether to evaluate synthetic data with another model")
    parser.add_argument(
        "--verifier-arch",
        type=str,
        default="mobilenet_v2",
        help="arch name from torchvision models to act as a verifier",
    )
    parser.add_argument("--ipc-start", default=0, type=int)
    parser.add_argument("--ipc-end", default=50, type=int)

    ## modifications_zx
    parser.add_argument('--init_path', default='/data/dujw/SREL/test/init/high_conf_50/', type=str)
    parser.add_argument('--init_part_num', default=0, type=int)


    args = parser.parse_args()

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    return args


if __name__ == "__main__":

    args = parse_args()
    args.milestone = 1

    # if not wandb.api.api_key:
    #     wandb.login(key='')
    # wandb.init(project='sre2l-cifar', name=args.exp_name)
    # global wandb_metrics
    # wandb_metrics = {}
    # for ipc_id in range(0,50):
    # cause we need n + n synthesize (ipc-1)*2
    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        # wandb.log({'ipc_id': ipc_id})
        main_syn(args, ipc_id)

    # wandb.finish()
