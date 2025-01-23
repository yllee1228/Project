import tensorflow as tf
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Eager execution 활성화
tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()

# tf.data 함수들도 강제로 eager 모드에서 실행되도록 설정
tf.data.experimental.enable_debug_mode()

# 현재 실행 모드 출력
print("Eager execution enabled:", tf.executing_eagerly())

def get_dataset_from_directory(root, subset, show_spec=True, batch_size=None, image_size=(256, 256)):
    dataset = image_dataset_from_directory(
        root,
        validation_split=0.2,
        subset=subset,
        seed=123,
        shuffle=True,
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