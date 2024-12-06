import argparse
from tqdm import tqdm
import torch
import numpy as np
from model import ResNet, Bottleneck
from utils import load_checkpoint, save_checkpoint, ensure_dir
from torch.utils.tensorboard import SummaryWriter
from dataloader import ImageDataset
import torchvision.transforms.v2 as transforms
from torch import nn
from torch.utils.data import DataLoader
import time
from matplotlib import pyplot as plt
from model import resnet50

def main(opt):
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu")
    print(f"Using {device} device")

    train_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=[3, 5], sigma=(0.1, 0.3))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomApply([transforms.GaussianNoise()], p=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )

    test_transforms = test_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )
    

    train_dataset = ImageDataset(image_dir = opt.train_dir, transform = train_transforms)
    test_dataset = ImageDataset(image_dir = opt.test_dir, transform = test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle=True, pin_memory=True,num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size = opt.batch_size, shuffle=False, pin_memory=True, num_workers=opt.num_workers)

    net = resnet50(pretrained=opt.pretrained, stride0=1, dropout=0.5).change_output(1).to(device)

    print("Number of parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optim = torch.optim.Adam(parameters, lr=opt.lr, betas=(0.9, 0.999))

    scaler = torch.GradScaler('cuda')

    start_n_iter = 0
    start_epoch = 0
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint) # custom method for loading last checkpoint
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']+1
        start_n_iter = ckpt['n_iter']
        optim.load_state_dict(ckpt['optim'])
        print("last checkpoint restored")

    writer = SummaryWriter()

    n_iter = start_n_iter
    best_accuracy = 0
    train_losses = []
    test_losses = []
    for epoch in range(start_epoch, opt.epochs):
        net.train()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        start_time = time.time()
        running_loss = 0.0
        for i, (images, labels) in pbar:
            optim.zero_grad()
            images = images.to(device)
            labels = labels.to(device).float()

            prepare_time = start_time - time.time()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = net(images).squeeze()
                loss = loss_fn(pred, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item() * images.size(0)

            process_time = start_time-time.time()-prepare_time
            compute_efficiency = process_time/(process_time+prepare_time)

            pbar.set_description(
                f'Compute efficiency: {compute_efficiency:.2f}, ' 
                f'loss: {loss.item():.2f},  epoch: {epoch}/{opt.epochs}')
            start_time = time.time()

            writer.add_scalar("Loss/Train", loss.item(), epoch*len(pbar) + i)

        train_loss = running_loss / len(train_dataset)
        train_losses.append(train_loss)

        if epoch % 1 == 0:
            net.eval()

            correct = 0
            total = 0

            pbar = tqdm(enumerate(test_dataloader),
                    total=len(test_dataloader)) 
            running_loss = 0.0
            fake_count = 0
            with torch.no_grad():
                for i, (images, labels) in pbar:
                    images, labels = images.to(device), labels.to(device).float()
                    
                    pred = net(images).squeeze()
                    loss = loss_fn(pred, labels)
                    running_loss += loss.item() * images.size(0)
                    predicted = (pred > 0.0).float()
                    fake_count += predicted.sum().item()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    writer.add_scalar("Loss/Test", loss.item(), epoch*len(pbar) + i)
            
            test_loss = running_loss / len(test_dataset)
            test_losses.append(test_loss)
            
            
            accuracy = correct/total
            writer.add_scalar("Accuracy/Test", accuracy, epoch)
            with open("accuracies.txt", 'a') as file:
                file.write(f'Epoch {epoch} - Accuracy on test set: {100*accuracy:.2f}\n')
            print(f'Epoch {epoch} - Accuracy on test set: {100*accuracy:.2f}')

            cpkt = {
                'net': net.state_dict(),
                'epoch': epoch,
                'n_iter': n_iter,
                'optim': optim.state_dict()
            }
            save_checkpoint(cpkt, f'model_checkpoint.ckpt', (accuracy>=best_accuracy))
            best_accuracy = max(best_accuracy, accuracy)

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(test_losses,label="val")
    plt.plot(train_losses,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('TrainValLoss.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        default = './AIGC-Detection-Dataset/AIGC-Detection-Dataset/train',
        type=str,
        help='Directory for train dataset'
    )
    parser.add_argument(
        '--test_dir',
        default = './AIGC-Detection-Dataset/AIGC-Detection-Dataset/val',
        type=str,
        help='Directory for test dataset'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Spcifies learing rate for optimizer. (default: 1e-4)')
    parser.add_argument(
        '--path_to_checkpoint',
        type=str,
        default='model_checkpoint.ckpt',
        help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs. (default: 10)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=12,
        help='Batch size for data loaders. (default: 12)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers for data loader. (default: 8)'
    )
    parser.add_argument(
        '--resume',
        action="store_true",
        help='Wether to resume the training from the stored checkpoint'
    )
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    main(args)
