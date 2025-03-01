import zmq
import time

def client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://<IP_MASTER_NODE>:5555")

    print("[INFO] GodNet-X Mesh Client Started...")

    for i in range(5):
        socket.send_string(f"Mesh Task {i}")
        message = socket.recv_string()
        print(f"[Mesh Client] Received: {message}")
        time.sleep(1)

if __name__ == "__main__":
    client()