import socket
import struct
import numpy as np
import threading


class Client:
    def __init__(self, ip='localhost', port=12345):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((ip, port))

    def send_array(self, array):
        array = array.astype(np.float64)
        array_bytes = array.tobytes()
        self.client.send(struct.pack(">L", len(array_bytes)))
        self.client.sendall(array_bytes)

    def receive_merged_array(self):
        data = b''
        payload_size = struct.calcsize(">L")
        while len(data) < payload_size:
            data += self.client.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += self.client.recv(4096)

        merged_array = np.frombuffer(data, dtype=np.float64)
        print(f"Merged array received: {merged_array}")


if __name__ == "__main__":
    client = Client()
    client_thread = threading.Thread(
        target=client.send_array, args=(np.random.rand(10),))
    client_thread.start()
    client_thread.join()  # Ensuring array is sent before receiving

    receive_thread = threading.Thread(target=client.receive_merged_array)
    receive_thread.start()
    receive_thread.join()  # Wait for merged array to be received
