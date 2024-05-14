from PIL import Image
import os


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


if __name__ == '__main__':
    _path = '../dataset/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
    _img = keep_image_size_open(_path)
    _img.show()
    masked_img = keep_image_size_open('../dataset/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png')
    masked_img.show()
    print(masked_img.size, _img.size)
    print(_img.size)