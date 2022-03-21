
import os
import albumentations as A
import glob
import numpy as np
import cv2 

printer_distortion = A.Compose([
    A.OneOf([
            A.CoarseDropout(max_holes=256, max_height=2, max_width=2, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
            A.CoarseDropout(max_holes=256, max_height=2, max_width=2, min_holes=None, min_height=None, min_width=None, fill_value=255, mask_fill_value=None, always_apply=False, p=0.5) ], p=0.9),
        A.RandomContrast(p = 0.8)
])
desktop_distortion = A.Compose([
    A.RandomContrast(p = 0.9),
    A.ShiftScaleRotate(p=0.9)
])
geometric_transform = A.Compose([
    A.RandomSunFlare(flare_roi=(0, 0, 0.3, 0.3), angle_lower=0.5, p=0.3),
    A.augmentations.geometric.transforms.Perspective(p = 0.8, scale = 0.05),
    A.augmentations.transforms.OpticalDistortion (distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=True, p=0.05),
    A.augmentations.transforms.GaussianBlur(),
    A.augmentations.transforms.Blur(p=0.4),
    A.transforms.ISONoise (color_shift=(0.01, 0.05), intensity=(0.1, 0.4), always_apply=False, p=0.4),
    #A.augmentations.transforms.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply= True, p=0.5)
    A.JpegCompression(quality_lower=40, quality_upper=100, always_apply=False, p=0.6)
])

# images from https://www.orgprint.com/wiki/lazernaja-pechat/defekty-lazernoj-pechati
desktop_backgrounds = glob.glob('desktop/*.*')
assert len(desktop_backgrounds) > 0

def blend_document_in_table(img, desktop_texture):
    h,w = img.shape[:2]
    desktop_h, desktop_w = desktop_texture.shape[:2]
    k = 1
    while k * desktop_h < h or k * desktop_w < w:
        k+=1
    desktop_h *= k
    desktop_w *= k
    desktop_texture = cv2.resize(desktop_texture, (desktop_w, desktop_h))
    
    prepared_im = printer_distortion(image=img)['image']
    dst = desktop_distortion(image=desktop_texture)['image']

#     mask = np.full(prepared_im.shape, 255, dtype = np.uint8)
#     center = ((desktop_w)//2, (desktop_h)//2)
#     return cv2.seamlessClone(prepared_im, dst, mask, center, cv2.NORMAL_CLONE)

    x,y = ((desktop_w-w)//2, (desktop_h-h)//2)
    dst[y:y+h,x:x+w] = prepared_im
    return dst

def augment_image(img):
    desktop_path = desktop_backgrounds[np.random.randint(len(desktop_backgrounds))]
    desktop_im = cv2.imread(desktop_path)
    img = blend_document_in_table(img, desktop_im)
    return geometric_transform(image=img)['image']


inputs = glob.glob('train/*.png')
os.mkdir('Generated')

for path in inputs:
    name = os.path.split(path)[-1]
    imname, ext = os.path.splitext(name)
    img = cv2.imread(path)
    for i in range(10):
        cv2.imwrite(f'Generated/{imname}_{i}{ext}', augment_image(img))





