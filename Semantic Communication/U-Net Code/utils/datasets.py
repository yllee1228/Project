import pickle, random
import numpy as np
import matplotlib.pyplot as plt
from config.train_config import SQRT_BATCH_SIZE
from config.train_config import DATA_TYPE
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf


# dir dataset loading 추가
def get_dataset_from_directory(root, subset, show_spec=True, batch_size=4, image_size=(256, 256)):
    dataset = image_dataset_from_directory(
        root,
        validation_split=0.2,
        subset=subset,
        seed=123,
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size,
        label_mode=None
    )

    dataset = dataset.map(lambda x: x / 255.0)

    if show_spec:
        # 데이터셋에서 첫 번째 배치만 가져옴
        for images_batch in dataset.take(1):
            # 이 배치에서 랜덤하게 하나의 이미지 선택
            random_image_index = random.randint(0, images_batch.shape[0] - 1)
            random_image = images_batch[random_image_index].numpy()
            
            # 선택된 이미지 정보 출력 및 표시
            print(f'length of dataset: {len(dataset)}')
            print('dataset image shape:', random_image.shape)
            print(f'Max pixel value: {np.max(random_image)}')
            print(f'Min pixel value: {np.min(random_image)}')
            plt.imshow(random_image)
            plt.show()

    # 이미지 모양 반환을 위한 첫 번째 배치의 첫 번째 이미지 사용
    image_shape = image_size + (3,)  # 이미지 크기와 채널 수를 바탕으로 이미지 모양 설정

    return dataset, image_shape


def getTrainDataset(root, show_spec=False):
    dataset = []
    for i in range(5):
        with open(root + f'/data_batch_{i+1}', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            dataset.append(np.reshape(dict[b'data'], (-1, 3, 32, 32)).transpose((0, 2, 3, 1)))
    dataset = np.vstack(dataset)

    if DATA_TYPE == 'CIFAR-10':
        dataset = dataset[0:int(dataset.shape[0]/SQRT_BATCH_SIZE**2)*SQRT_BATCH_SIZE**2]
        dataset = np.reshape(dataset, (-1, SQRT_BATCH_SIZE, SQRT_BATCH_SIZE, 32, 32, 3))
        dataset = np.transpose(dataset, (0, 1, 3, 2, 4, 5))
        dataset = np.reshape(dataset, (-1, 1, 32*SQRT_BATCH_SIZE, 32*SQRT_BATCH_SIZE, 3))
        
    if show_spec:
        print('length of dataset: ', len(dataset))
        print('dataset image shape:', dataset[0][0].shape)
        print('example image:')
        plt.imshow(dataset[0][0])
        plt.show()
    
    return tf.data.Dataset.from_tensor_slices(tf.cast(dataset, tf.float32)), dataset[0][0].shape


def getTestDataset(root, show_spec=False):
    with open(root + f'/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        dataset = np.reshape(dict[b'data'], (-1, 3, 32, 32)).transpose((0, 2, 3, 1))

    if DATA_TYPE == 'CIFAR-10':
        dataset = dataset[0:int(dataset.shape[0]/SQRT_BATCH_SIZE**2)*SQRT_BATCH_SIZE**2]
        dataset = np.reshape(dataset, (-1, SQRT_BATCH_SIZE, SQRT_BATCH_SIZE, 32, 32, 3))
        dataset = np.transpose(dataset, (0, 1, 3, 2, 4, 5))
        dataset = np.reshape(dataset, (-1, 1, 32*SQRT_BATCH_SIZE, 32*SQRT_BATCH_SIZE, 3))
        
    if show_spec:
        print('length of dataset: ', len(dataset))
        print('dataset image shape:', dataset[0][0].shape)
        print('example image:')
        plt.imshow(dataset[0][0])
        plt.show()
    
    return tf.data.Dataset.from_tensor_slices(tf.cast(dataset, tf.float32)), dataset[0][0].shape