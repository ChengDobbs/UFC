import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb

from imagenet_ipc_index import ImageFolderIPC

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Post-Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--output-dir", default="./save", type=str)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--check-ckpt", default=None, type=str)
parser.add_argument("--batch-size", default=128, type=int)

parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--syn-data-path", default="/data/dujw/SREL/test/cifar100_conf25+1universal+multimodel+bn/generated_results", type=str)
parser.add_argument("--teacher-path", default="/home/dujw/Robustness/otho_SRE/codes/save/cifar100/resnet18_E200/ckpt.pth", type=str)
parser.add_argument("--ipc", default=50, type=int)

parser.add_argument('--wandb-project', type=str,
                    default='final_rn18_fkd', help='wandb project name')
parser.add_argument('--wandb-api-key', type=str,
                    default=None, help='wandb api key')
parser.add_argument('--wandb-name', type=str,
        default="db_name", help='name')
 
args = parser.parse_args()

#init wandb 
from datetime import datetime
date_time = datetime.now().strftime("%m/%d, %H:%M:%S")
wandb.login(key=args.wandb_api_key)
wandb.init(project=args.wandb_project, name=args.wandb_name+"_"+date_time)



if args.check_ckpt:
    checkpoint = torch.load(args.check_ckpt)
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print(f"==> test ckp: {args.check_ckpt}, acc: {best_acc}, epoch: {start_epoch}")
    exit()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

print("=> Using IPC setting of ", args.ipc)
trainset = ImageFolderIPC(root=args.syn_data_path, transform=transform_train, ipc=args.ipc)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root="../data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Model
print("==> Building model..")

model = torchvision.models.resnet18(num_classes=100)
# model = torchvision.models.get_model("resnet18", num_classes=100)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()


model_student = model.to(device)
if device == "cuda":
    model_student = torch.nn.DataParallel(model_student)
    cudnn.benchmark = True

# model_teacher = torchvision.models.get_model("resnet18", num_classes=100)
model_teacher = torchvision.models.resnet18( num_classes=100)
model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model_teacher.maxpool = nn.Identity()

model_teacher = nn.DataParallel(model_teacher).cuda()
# teacher model load from checkpoint
checkpoint = torch.load(args.teacher_path)
model_teacher.load_state_dict(checkpoint["state_dict"])

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    model_student.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_student.parameters(), lr=0.001, weight_decay=0.01)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
args.temperature = 30
loss_function_kl = nn.KLDivLoss(reduction="batchmean")


def mixup_data(x, y, index_ipc_cls, alpha=0.8):
    """
    Returns mixed inputs, mixed targets, and mixing coefficients.
    For normal learning
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    while True:
        img_a = index_ipc_cls
        img_b = [index_ipc_cls[i] for i in index]
        condition = any(img_a[i][1] == img_b[i][1] and int(img_a[i][1]) % 48 == int(img_b[i][1]) % 48 for i in range(batch_size))
        if not condition:
            break
        index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Train
def train(epoch,wandb_metrics):
    model_student.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, paths, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        index_ipc_cls = [(path.split('id')[1].split('.')[0], path.split('class')[1].split('_')[0]) for path in paths]

        inputs, target_a, target_b, lam = mixup_data(inputs, targets, index_ipc_cls)

        optimizer.zero_grad()
        outputs = model_student(inputs)
        outputs_ = F.log_softmax(outputs / args.temperature, dim=1)

        # teacher model pretrained, params load from checkpoint (saved in squeeze phase)
        soft_label = model_teacher(inputs).detach()
        soft_label_ = F.softmax(soft_label / args.temperature, dim=1)

        # crucial to make synthetic data and labels more aligned
        loss = loss_function_kl(outputs_, soft_label_)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Epoch: [{epoch}], Acc@1 {100.*correct/total:.3f}, Loss {train_loss/(batch_idx+1):.4f}")
    metrics = {
        "train/loss": float(f"{train_loss/(batch_idx+1):.4f}"),
        "train/Top1": float(f"{100.*correct/total:.3f}"),
        "train/epoch": epoch,}
    wandb_metrics.update(metrics)
    wandb.log(wandb_metrics)


# Test
def test(epoch,wandb_metrics):
    global best_acc
    model_student.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_student(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Test: Acc@1 {100.*correct/total:.3f}, Loss {test_loss/(batch_idx+1):.4f}")

    acc = 100.0 * correct / total
    if acc > best_acc:
        best_acc = acc

    metrics = {
        'val/loss': float(f"{test_loss/(batch_idx+1):.4f}"),
        'val/top1': float(f"{100.*correct/total:.3f}"),
        'val/epoch': epoch,
        'val/best_acc':best_acc,
    }
    wandb_metrics.update(metrics)
    wandb.log(wandb_metrics)



    # Save checkpoint.
    # save last checkpoint
    if True:
        state = {
            "state_dict": model_student.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')

        path = os.path.join(args.output_dir, "./ckpt.pth")
        torch.save(state, path)
        # best_acc = acc


start_time = time.time()
for epoch in range(start_epoch, start_epoch + args.epochs):
    global wandb_metrics
    wandb_metrics = {}

    train(epoch, wandb_metrics)
    # fast test
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        test(epoch, wandb_metrics)
    scheduler.step()
end_time = time.time()
wandb.finish()
print(f"total time: {end_time - start_time} s")

