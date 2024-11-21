import socket
# from dash import clientside_callback
import rospy
from std_msgs.msg import String

# SERVER_IP = '10.1.80.81'
SERVER_IP = '0.0.0.0'
PORT = 5000

def main():
    rospy.init_node('RT_audio_classification_pub', anonymous=True)
    pub = rospy.Publisher('work_situation_class', String, queue_size=10)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind((SERVER_IP, PORT))
        server_socket.listen(1)
        print("Server is ready :)")
    except Exception as e:
        print(f"Error while start server: {e}")

    print("Standby for client connect...")

    conn, addr = server_socket.accept()
    print(f"Client connected: {addr}")

    try:
        while not rospy.is_shutdown():
            data = conn.recv(1024)
            if not data:
                break
            received_data = data.decode('utf-8')
            rospy.loginfo(f"Received data: {received_data}")

            pub.publish(received_data)

    except KeyboardInterrupt:
        print("Server is shutting down...")

    finally:
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    main()