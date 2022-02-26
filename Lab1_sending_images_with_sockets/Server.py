import cv2
import socket
import numpy as np

from ImageSender import ImageSender


class Server(ImageSender):
    def __init__(self, host, server_port, block_size=1024):
        super().__init__(block_size, 'Server')
        self.port = server_port
        self.host = host
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, server_port))
        self.socket.listen()
        self.original_image = None
        self.image_with_noise = None
        self.image_without_noise = None

    def run(self):
        with self.socket as s:
            self.log(f'Подключение к порту {self.port}')
            conn, _ = s.accept()
            self.log('Подключено. Получение изображения.')
            self.original_image = self.recv_image(conn)
            self.image_with_noise = self.recv_image(conn)

        self.image_without_noise = self.denoise(self.image_with_noise)

        self.log('Сравнение оригинала с зашумленным изображением')
        self.compare_images(self.original_image, self.image_with_noise)
        self.log('Сравнение оригинала с восстановленным изображением')
        self.compare_images(self.original_image, self.image_without_noise)

        cv2.imwrite('images/result.jpg', self.image_without_noise)
        cv2.imwrite('images/noise.jpg', self.image_with_noise)

    def compare_images(self, image1, image2):
        diff = (image1 - image2).astype('float32')
        metric = np.abs(diff).mean()
        self.log(f'Среднее абсолютное отклонение: {metric}')
        metric = np.square(diff).mean()
        self.log(f'Среднеквадратичное отклонение: {metric}')

    @staticmethod
    def denoise(image):
        return cv2.medianBlur(image, ksize=5)


if __name__ == '__main__':
    server = Server(host='127.0.0.1', server_port=1001)
    server.start()
