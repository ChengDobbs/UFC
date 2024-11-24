python relabel_cifar.py \
    --epochs 800 \
    --output-dir '../save/post_cifar100/ipc50/E800_3060' \
    --syn-data-path '../syn_data/cifar100_rn18_1K_mobile.lr0.25.bn0.01' \
    --teacher-path '../save/cifar100/resnet18_E200/ckpt.pth' \
    --ipc 50
    --batch-size 128

