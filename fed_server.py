import socket
import struct
import threading
import numpy as np

from Models import Unet
from Models import AttnUnet
from Models import ResUnet

global_size = (128, 128, 3)


class Server:
    def __init__(self, ip='localhost', port=12345):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((ip, port))
        self.server.listen(5)
        self.clients = []
        self.arrays = []
        self.unet_global = Unet((global_size))
        self.global_model = self.unet_global.build()

    def start(self):
        print("Server is listening for connections...")
        while len(self.clients) < 3:
            conn, addr = self.server.accept()
            print("Connection established from ", addr)
            self.clients.append(conn)
            threading.Thread(target=self.handle_client,
                             args=(conn, len(self.clients))).start()

    def handle_client(self, conn, client_num):
        data = b''
        payload_size = struct.calcsize(">L")
        while True:
            while len(data) < payload_size:
                data += conn.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)

            array_received = np.frombuffer(data[:msg_size], dtype=np.float64)
            data = data[msg_size:]
            print(f'Received array {array_received} from client {client_num}')

            self.arrays.append(np.array(array_received))
            if len(self.arrays) == 3:
                self.merge_and_send_arrays()

    def merge_and_send_arrays(self):
        # merged_array = np.concatenate(self.arrays)
        # weights = [0.1,0.1,0.1,0.1]
        weighted_avg = np.average(self.arrays, axis=0)  # weights=weights
        # merged_array_bytes = merged_array.tobytes()
        self.global_model.set_weights(weighted_avg)
        self.global_model.save('global_model.h5')
        merged_array_bytes = weighted_avg.tobytes()
        for client in self.clients:
            client.send(struct.pack(">L", len(merged_array_bytes)))
            client.sendall(merged_array_bytes)


if __name__ == "__main__":
    server = Server()
    threading.Thread(target=server.start).start()
