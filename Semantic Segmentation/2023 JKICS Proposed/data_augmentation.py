import os
import numpy as np
from patchify import patchify, unpatchify
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A


def augment(width, height): #증강 함수
    transform = A.Compose([
        A.RandomCrop(width=width, height=height, p=1.0), #무작위 자르기
        A.HorizontalFlip(p=1.0), # 수평 뒤집기
        A.VerticalFlip(p=1.0), # 수직 뒤집기
        A.Rotate(limit=[60, 300], p=1.0, interpolation=cv2.INTER_NEAREST), # 회전
        #A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.3], contrast_limit=0.2, p=1.0), #무작위 밝기 및 대비
        A.OneOf([
            A.CLAHE (clip_limit=1.5, tile_grid_size=(8, 8), p=0.5), #적응형 히스토그램 균등화(CLAHE) 
            A.GridDistortion(p=0.5), # 그리드 왜곡
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5), #광학 왜곡
        ], p=1.0),
    ], p=1.0)
    
    return transform


def visualize(image, mask, original_image=None, original_mask=None): #시각화 함수
    fontsize = 16

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(10, 10), squeeze=True)
        f.set_tight_layout(h_pad=5, w_pad=5)

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 12), squeeze=True)
        plt.tight_layout(pad=0.2, w_pad=1.0, h_pad=0.01)

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original Image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original Mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed Image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed Mask', fontsize=fontsize)
        
    plt.savefig('./test/sample_augmented_image.png', facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 100)


# 데이터 확인
# ##########################################################################################
# image = cv2.imread("./dataset/image/image_part_1 (1).jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# mask = cv2.imread("./dataset/mask/image_part_1 (1).png")
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

# transform = augment(512, 512)
# transformed = transform(image=image, mask=mask)
# transformed_image = transformed['image']
# transformed_mask = transformed['mask']

# cv2.imwrite('./test/image_test.png',cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
# cv2.imwrite('./test/mask_test.png',cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2RGB))

# visualize(transformed_image, transformed_mask, image, mask)


# Albumentations 라이브러리를 사용한 데이터 증강
############################################################################################

images_dir = './Image/image512/'
masks_dir = './Image/mask512/'

file_names = np.sort(os.listdir(images_dir)) 
file_names = np.char.split(file_names, '.')
filenames = np.array([])
for i in range(len(file_names)):
    filenames = np.append(filenames, file_names[i][0])
    

def augment_dataset(count): #데이터 증강 함수
    transform_1 = augment(256, 256)
    # transform_1 = augment(256, 256)
    
    i = 0
    for i in range(count):
        for file in filenames:
            img = cv2.imread(images_dir+file+'.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(masks_dir+file+'.png')
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            transformed = transform_1(image=img, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

                
            cv2.imwrite('./datasets/aug_dataset/aug_image/aug_{}_'.format(str(i+1))+file+'.jpg',cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
            cv2.imwrite('./datasets/aug_dataset/aug_mask/aug_{}_'.format(str(i+1))+file+'.png',cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2RGB))
            

# augment_dataset(16)

input_dir="./datasets/aug_dataset/aug_image"        # Input directory for images
mask_dir="./datasets/aug_dataset/aug_mask512"          # Input directory for masks
output_image_dir="./datasets/aug_dataset/aug_image512"   # Output directory for resized images
output_mask_dir="./datasets/aug_dataset/aug_mask"      # Output directory for resized masks

resize_shape = (512, 512)

# image resizing code
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

for file in os.listdir(input_dir):
    try:
        file_path = os.path.join(input_dir, file)
        if not file.endswith(".jpg") and not file.endswith(".png"):
            print(f"Skipping unsupported file: {file}")
            continue

        # 파일 읽기
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Cannot read file: {file_path}")

        # 파일 리사이징 및 저장
        resized_img = cv2.resize(img, (512, 512))
        output_path = os.path.join(output_image_dir, file)
        cv2.imwrite(output_path, resized_img)
        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Skipping file {file}: {e}")

# mask resizing code
# if not os.path.exists(output_mask_dir):
#     os.makedirs(output_mask_dir)

# for file in os.listdir(mask_dir):
#     try:
#         file_path = os.path.join(mask_dir, file)
#         if not file.endswith(".jpg") and not file.endswith(".png"):
#             print(f"Skipping unsupported file: {file}")
#             continue

#         # 파일 읽기
#         img = cv2.imread(file_path)
#         if img is None:
#             raise ValueError(f"Cannot read file: {file_path}")

#         # 파일 리사이징 및 저장
#         resized_img = cv2.resize(img, (256, 256))
#         output_path = os.path.join(output_mask_dir, file)
#         cv2.imwrite(output_path, resized_img)
#         print(f"Processed and saved: {output_path}")

#     except Exception as e:
#         print(f"Skipping file {file}: {e}")

