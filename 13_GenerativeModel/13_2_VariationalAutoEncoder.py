# tensorflow_probability 가 tf25 요구함
# 가상환경 tf25 사용

"""
chapter 13. 생성 모델
    13.2 변형 오토인코더
"""

'''
2. 변형 오토인코더 (Variational Autoencoder)
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pandas as pd
import tensorflow_probability as tfp
ds = tfp.distributions


# GPU 작업 중복 문제 - 싱글 gpu 사용 시 해결 방안
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# 초깃값 설정
TRAIN_BUF = 60000
TEST_BUF = 10000
BATCH_SIZE = 512
N_TRAIN_BATCHES = int(TRAIN_BUF/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)


# fashion_mnist 데이터셋 로딩 및 데이터셋 분리
(train_images, train_lebels), (test_images, test_lebels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0

train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(512)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_images).shuffle(10000).batch(512)
)


# 변형 오토인코더 네트워크 생성
class VAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    def encode(self, x):
        mu, sigma = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def reparameterize(self, mean, logvar):
        epsilon = tf.random.normal(shape=mean.shape)
        return epsilon * tf.exp(logvar * 0.5) + mean

    def recontruct(self, x):
        mu, _ = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return self.decode(mu)

    def decode(self, z):
        return self.dec(z)

    def loss_function(self, x):
        q_z = self.encode(x)
        z = q_z.sample()
        x_recon = self.decode(z)
        p_z = ds.MultivariateNormalDiag(
            loc=[0.] * z.shape[-1], scale_diag=[1.] * z.shape[-1]
        )
        kl_div = ds.kl_divergence(q_z, p_z)
        latent_loss = tf.reduce_mean(tf.max(kl_div,0))
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(x-x_recon), axis=0))
        return recon_loss, latent_loss

    def gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x)
        return tape.gradient(loss, self.trainable_variables)

    @tf.function
    def train(self, train_x):
        gradients = self.gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))