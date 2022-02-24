import socket
from random import random

from ImageSender import ImageSender


class Noise(ImageSender):
    def __init__(self, host, noise_port, server_port, block_size=1024, noise_probability=0.1):
        super().__init__(block_size, 'Noise')
        # Receives image from the client
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.recv_socket.bind((host, noise_port))
        self.recv_socket.listen()
        self.noise_port = noise_port

        # Sends image to the server
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_port = server_port

        self.noise_probability = noise_probability

    def run(self):
        with self.recv_socket as s:
            self.log(f'Подключение к порту {self.noise_port}')
            conn, _ = s.accept()
            self.log('Подключено. Получение изображения.')
            image = self.recv_image(conn)

        noise_image = self.noise(image, self.noise_probability)
        with self.send_socket as s:
            s.connect((self.host, self.server_port))
            self.send_image(image, s)
            self.send_image(noise_image, s)

    @staticmethod
    def noise(img, noise_prob=0.1):
        img = img.copy()
        height, width, _ = img.shape
        for y in range(height):
            for x in range(width):
                if random() < noise_prob:
                    img[y, x] = int(random() > 0.5) * 255
        return img


if __name__ == '__main__':
    server = Noise(
        host='localhost_lab1',
        noise_port=1000,
        server_port=1001,
        noise_probability=0.1
    )
    server.start()
