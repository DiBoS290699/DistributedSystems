from threading import Thread
import numpy as np

class ImageSender(Thread):
    """
    Класс, передающий и принимающий изображения с помощью сокета
    """
    def __init__(self, block_size=1024, name='ImageSender'):
        super().__init__()
        self.block_size = block_size
        self.name = name

    def log(self, msg):
        print(f'{self.name}:', msg)

    def send_image(self, image, _socket):
        n_bytes = self.send_image_props(image, _socket)
        n_blocks, remains = divmod(n_bytes, self.block_size)
        image_bytes = image.tobytes()
        for i in range(n_blocks):
            block = image_bytes[self.block_size * i: self.block_size * (i + 1)]
            if _socket.sendall(block) is not None:
                self.log(f'Ошибка при передаче байтов изображения в блоке i={i}')

        if remains > 0:
            block = image_bytes[-remains:]
            if _socket.sendall(block) is not None:
                self.log(f'Ошибка при передаче остатка байтов изображения')
        self.log('Изображение отправлено.')

    def send_image_props(self, image, _socket):
        """
        Отправка параметров (высоты, ширины, кол-ва байтов) изображения

        :param image: Изображение для отправки
        :param _socket: Сокет
        :return: Кол-во байт изображения
        """
        height, width, n_colors = image.shape
        n_bytes = height * width * n_colors

        if self.send_int(n_bytes, _socket) is not None:
            self.log('Ошибка передачи общего количества байтов')
        if self.send_int(height, _socket) is not None:
            self.log('Ошибка передечи высоты изображения')
        if self.send_int(width, _socket) is not None:
            self.log('Ошибка передечи ширины изображения')
        return n_bytes

    @staticmethod
    def send_int(number, _socket):
        """
        Конвертация числа в байты с ограничением до 4 байтов
        со значимым байтом в начале массива

        :param number: Конвертируемое число
        :param _socket: Сокет
        :return: None, если выполнено удачно
        """
        _bytes = number.to_bytes(length=4, byteorder='big')
        return _socket.sendall(_bytes)

    def recv_image(self, conn):
        n_bytes, height, width = self.recv_image_props(conn)
        im_bytes = []
        n_received = 0
        while n_received != n_bytes:
            block = conn.recv(self.block_size)
            im_bytes.append(block)
            n_received += len(block)

        im_bytes = b''.join(im_bytes)
        image = np.frombuffer(im_bytes, dtype='uint8')
        image = image.reshape(height, width, 3)        # 3 цвета (RGB)
        self.log(f'Принято изображение размером {height}x{width} и кол-вом байт {n_bytes}')
        return image

    def recv_image_props(self, conn):
        n_bytes = self.recv_int(conn)
        height = self.recv_int(conn)
        width = self.recv_int(conn)
        return n_bytes, height, width

    @staticmethod
    def recv_int(conn):
        _bytes = conn.recv(4)
        return int.from_bytes(_bytes, byteorder='big')
