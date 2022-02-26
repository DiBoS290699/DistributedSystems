import socket
import cv2

from ImageSender import ImageSender


class Client(ImageSender):
    def __init__(self, host, server_port, image, block_size=1024):
        super().__init__(block_size, 'Client')
        self.port = server_port
        self.image = image
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host

    def run(self):
        with self.socket as s:
            self.log(f'Подключение к порту {self.port}')
            s.connect((self.host, self.port))
            self.log('Подключено. Отправка изображения.')
            self.send_image(self.image, s)


if __name__ == '__main__':
    img = cv2.imread('images/original.jpg')
    client = Client(host='127.0.0.1', server_port=1000, image=img)
    client.start()
