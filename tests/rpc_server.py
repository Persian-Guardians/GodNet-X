import zmq

def server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("[INFO] Persian Guardians RPC Server Running...")

    while True:
        message = socket.recv_string()
        print(f"[RPC Server] Received: {message}")
        socket.send_string("ACK")

if __name__ == "__main__":
    server()