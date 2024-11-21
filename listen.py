import socket

from dash import clientside_callback

# SERVER_IP = '10.1.80.81'
SERVER_IP = '0.0.0.0'
PORT = 5000
try:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, PORT))
    server_socket.listen(1)
    print("Server is ready :)")
except Exception as e:
    print(f"Error while start server: {e}")

print("server ready. standby for client connect...")

conn, addr = server_socket.accept()
print(f"Client connected: {addr}")

try:
    while True:
        data = conn.recv(1024)
        if not data:
            break
        print(f"Received data: {data.decode('utf-8')}")
except KeyboardInterrupt:
    print("Server is shutting down...")

finally:
    conn.close()
    server_socket.close()
