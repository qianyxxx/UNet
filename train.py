import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from model import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_path = 'params/unet.pth'
data_path = '../dataset/VOCdevkit/VOC2012'
save_path = 'train_image'

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=2, shuffle=True)
    net = UNet().to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('success load weight')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    epoch = 1
    while epoch < 100:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()  # 更新梯度

            if i % 5 == 0:
                print(f'{epoch}--{i}-train_loss===>>{train_loss.item()}')

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            # _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            # _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{epoch}-{i}.png')

            if epoch % 20 == 0:
                torch.save(net.state_dict(), weight_path)
                print('save model successfully!')

        epoch += 1
