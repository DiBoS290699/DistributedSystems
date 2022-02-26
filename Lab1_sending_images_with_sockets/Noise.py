import socket
from random import random, randint
from time import time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from ImageSender import ImageSender


class Noise(ImageSender):
    def __init__(self, host, noise_port, server_port, block_size=1024, noise_probability=0.1, use_numba=False):
        super().__init__(block_size, 'Noise')

        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.recv_socket.bind((host, noise_port))
        self.recv_socket.listen()
        self.noise_port = noise_port
        self.use_numba = use_numba      # Использование параллельного алгоритма генерации шума

        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_port = server_port

        self.noise_probability = noise_probability

    def run(self):
        with self.recv_socket as s:
            self.log(f'Подключение к порту {self.noise_port}')
            conn, _ = s.accept()
            self.log('Подключено. Получение изображения.')
            image = self.recv_image(conn)

        if self.use_numba and cuda.is_available():
            self.log('Начало параллельной генерации шума')
            start_time = time()
            height, width, _ = image.shape
            grid_x_dim = int((width + 31) / 32)
            grid_y_dim = int((height + 31) / 32)
            states = create_xoroshiro128p_states(n=height * width, seed=randint(0, 100))
            dev_image = cuda.to_device(image)
            gpu_noise[(grid_x_dim, grid_y_dim), (32, 32)](dev_image, self.noise_probability, states, height, width)
            noise_image = dev_image.copy_to_host()
            noising_time = time() - start_time
            self.log(f'Параллельная генерация шума завершена за {noising_time} секунд')
        else:
            self.log('Начало последовательной генерации шума')
            start_time = time()
            noise_image = self.cpu_noise(image, self.noise_probability)
            noising_time = time() - start_time
            self.log(f'Последовательная генерация шума завершена за {noising_time} секунд')
        with self.send_socket as s:
            s.connect((self.host, self.server_port))
            self.send_image(image, s)
            self.send_image(noise_image, s)

    @staticmethod
    def cpu_noise(img, noise_prob=0.1):
        img = img.copy()
        height, width, _ = img.shape
        for y in range(height):
            for x in range(width):
                if random() < noise_prob:
                    img[y, x] = int(random() > 0.5) * 255
        return img


@cuda.jit
def gpu_noise(img, noise_prob, states, height, width):
    x, y = cuda.grid(2)
    n_state = x * y
    if x < width and y < height and xoroshiro128p_uniform_float32(states, n_state) < noise_prob:
        img[y, x] = int(xoroshiro128p_uniform_float32(states, n_state) > 0.5) * 255


if __name__ == '__main__':
    server = Noise(
        host='localhost_lab1',
        noise_port=1000,
        server_port=1001,
        noise_probability=0.1
    )
    server.start()
