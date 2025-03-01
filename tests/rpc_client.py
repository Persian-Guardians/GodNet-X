import zmq
import time

def client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    print("[INFO] Persian Guardians RPC Client Started...")

    for i in range(5):
        socket.send_string(f"Ping {i}")
        message = socket.recv_string()
        print(f"[RPC Client] Received: {message}")
        time.sleep(1)

if __name__ == "__main__":
    client()