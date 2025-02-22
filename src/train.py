import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

from args import get_parser
from dataloader.cityscapes import CityScapesDataset
from models.generator import SPADEGenerator
from models.discriminator import SPADEDiscriminator
from models.ganloss import GANLoss
from models.weights_init import weights_init

from torchvision import transforms
from torchvision.datasets import VOCSegmentation

from models.novograd import NovoGrad

def check_grad(parameters):
    sum_grad = 0
    i = 0
    for p in parameters:
        if p.grad is not None:
            sum_grad = p.grad.abs().mean()
            i += 1
    return sum_grad / i


def train(args):
    # Get the data
    # path = args.path
    # dataset = {
    #     x: CityScapesDataset(path, split=x, is_transform=True, img_size=args.img_size) for x in ['train', 'val']
    # }
    # data = {
    #     x: torch.utils.data.DataLoader(dataset[x],
    #                                    batch_size=args.batch_size,
    #                                    shuffle=True,
    #                                    num_workers=args.num_workers,
    #                                    drop_last=True) for x in ['train', 'val']
    # }

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
       transforms.Resize((256, 256)),
       transforms.ToTensor()
    ])
    # test_transform = transforms.Compose([
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     transforms.ToTensor()
    # ])

    trainset = VOCSegmentation("./data", year='2012', image_set='train', download=True, transform=train_transform, target_transform=target_transform)
    # testset = VOCSegmentation("./data", year='2012', image_set='val', download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    epochs = args.epochs
    lr_gen = args.lr_gen
    lr_dis = args.lr_dis
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise Exception('GPU not available')
    torch.backends.cudnn.benchmark = True

    gen = SPADEGenerator(args)
    dis = SPADEDiscriminator(args)

    gen = gen.to(device)
    dis = dis.to(device)

    criterion = GANLoss()

    gen.apply(weights_init)
    dis.apply(weights_init)

    optim_gen = NovoGrad(gen.parameters(), lr=lr_gen)
    optim_dis = NovoGrad(dis.parameters(), lr=lr_dis)

    img_lists = []
    G_losses = []
    D_losses = []

    # The training loop
    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch+1}/{epochs}')
        gen_loss = 0
        dis_loss = 0
        gen_grad = 0
        dis_grad = 0
        gen.train()
        dis.train()
        for i, (img, seg) in enumerate(train_loader):

            noise = torch.tensor(np.random.randn(args.batch_size, 256), dtype=torch.float32, requires_grad=True)
            noise = noise.to(device)
            img = img.to(device)
            seg = seg.to(device)
            
            fake_img = gen(noise, seg)
            pred_fake = dis(fake_img, seg)
            loss_G = criterion(pred_fake, True, generator=True)
            optim_gen.zero_grad()
            loss_G.backward()
            optim_gen.step()
            gen_grad += check_grad(gen.parameters())


            fake_img = fake_img.detach()
            pred_fake = dis(fake_img, seg)
            loss_D_fake = criterion(pred_fake, False)
            pred_real = dis(img, seg)
            loss_D_real = criterion(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            optim_dis.zero_grad()
            loss_D.backward()
            optim_dis.step()

            G_losses.append(loss_G.detach().cpu())
            D_losses.append(loss_D.detach().cpu())
            gen_loss += loss_G.detach().cpu().item()
            dis_loss += loss_D.detach().cpu().item()
            dis_grad += check_grad(dis.parameters())

        print()
        if epoch % 10 == 0:
            with torch.no_grad():
                im = np.clip((fake_img[0].detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)
                im = im.transpose((1, 2, 0))
                im = Image.fromarray(im, "RGB")
                im.save(f"imgs/img-e{epoch:04}.jpg")
            torch.save(gen, 'gen.pth')
            torch.save(gen, 'dis.pth')


        print("Gen loss", gen_loss / len(train_loader), "Gen Grad", gen_grad / len(train_loader), "Dis loss", dis_loss / len(train_loader), "Dis grad", dis_grad / len(train_loader))



if __name__ == "__main__":
    # Parse the arguments
    parser = get_parser()
    args = parser.parse_args()

    if args.gen_hidden_size % 16 != 0:
        print('hidden-size should be multiple of 16. It is based on paper where first input', end=" ")
        print('to SPADE is (4,4) in height and width. You can change this defualt in args.py')
        exit()

    args.img_size = tuple(args.img_size)

    train(args)
