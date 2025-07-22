import zmq
import numpy as np
from threading import Thread
from queue import Queue, Empty
import time

# ===========================
# Publisher (Server) Section
# ===========================
class PubServer:
    def __init__(self, host="*", port=5556):
        self.ctx = zmq.Context(io_threads=2)
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.setsockopt(zmq.SNDHWM, 10)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 30)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
        self.socket.bind(f"tcp://{host}:{port}")
        self.queue = Queue(maxsize=1)
        self.thread = Thread(target=self._publisher_loop, daemon=True)
        self.sent_counter = 0
        self.start_time = time.time()

    def _serialize_array(self, arr):
        """Convert numpy array to C-style bytes with header"""
        header = np.array(arr.shape, dtype=np.uint32).tobytes()
        data = arr.astype(np.float32).tobytes(order='C')
        return header + data

    def _publisher_loop(self):
        while True:
            if not self.queue.empty():
                arr = self.queue.get()
                self.socket.send(self._serialize_array(arr))

    def publish(self, array):
        if not self.queue.full():
            self.queue.put(array)

    def start(self):
        self.thread.start()

# ===========================
# Subscriber (Client) Section
# ===========================
class SubClient:
    def __init__(self, server_ip, port=5556):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 10)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 30)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.socket.connect(f"tcp://{server_ip}:{port}")
        self.latency_samples = []

    def receive(self, timeout_ms=1):
        """Non-blocking receive with polling and latency measurement."""
        data = self.socket.recv()  # This call blocks until data arrives[3][5][7][8]
        header = np.frombuffer(data[:8], dtype=np.uint32)
        arr = np.frombuffer(data[8:], dtype=np.float32).reshape(header)
        return arr

    # In stream_utils.py - update deserialization to handle 1D arrays
    def _deserialize(self, data):
        header = np.frombuffer(data[:8], dtype=np.uint32)
        return np.frombuffer(data[8:], dtype=np.float32).reshape(header)

