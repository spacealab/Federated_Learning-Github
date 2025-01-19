import socket
import pickle
import time
import psutil
from yolov8_model import train_local_model, update_model_weights

def limit_cpu_usage(target_percentage):
    """
    Limit CPU usage to the specified target percentage.
    """
    process = psutil.Process()
    while True:
        cpu_usage = process.cpu_percent(interval=0.1)
        if cpu_usage > target_percentage:
            time.sleep(0.05)  # Introduce a small delay to reduce CPU usage
        else:
            break

def connect_to_server(port):
    """
    Connect to the server on the specified port.
    """
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(30)  # Set timeout for connection (30 seconds)
        client_socket.connect(("127.0.0.1", port))
        print(f"Connected to server on port {port}.")
        return client_socket
    except Exception as e:
        print(f"Error connecting to server on port {port}: {e}")
        return None

def send_weights_to_server(client_socket, weights):
    """
    Send local weights to the server.
    """
    try:
        client_socket.sendall(pickle.dumps(weights))  # Use sendall for reliable transfer
        print("Weights sent to server.")
    except Exception as e:
        print(f"Error sending weights to server: {e}")
        raise e  # Re-raise the exception to stop further execution

def receive_weights_from_server(port):
    """
    Receive global weights from the server on the specified port.
    """
    try:
        client_socket = connect_to_server(port)
        if client_socket is None:
            raise ValueError("Failed to connect to server.")

        data = b""
        while True:
            packet = client_socket.recv(4096)  # Receive in chunks
            if not packet:
                break
            data += packet
        if not data:
            raise ValueError("No data received from server.")
        global_weights = pickle.loads(data)
        print("Received global weights from server.")
        return global_weights
    except Exception as e:
        print(f"Error receiving weights from server: {e}")
        raise e  # Re-raise the exception to stop further execution
    finally:
        client_socket.close()

def save_weights_to_file(weights, filename):
    """
    Save weights to a file.
    """
    try:
        with open(filename, "wb") as f:
            pickle.dump(weights, f)
        print(f"Weights saved to {filename}.")
    except Exception as e:
        print(f"Error saving weights to file: {e}")

if __name__ == "__main__":
    # Limit CPU usage before training
    print("Limiting CPU usage to 45%...")
    limit_cpu_usage(target_percentage=45)

    # Connect to the server (port 9090) to send local weights
    client_socket = connect_to_server(9090)
    if client_socket is None:
        print("Failed to connect to the server. Exiting.")
        exit(1)

    # YOLOv8 data path for Client 1
    local_data_path = "datasets/client1/data.yaml"  # Path specific to Client 1

    # Train the local model
    try:
        print("Training local model...")
        local_weights = train_local_model(local_data_path)
        # Save local weights to a file
        save_weights_to_file(local_weights, "local_weights_client1.pkl")
    except Exception as e:
        print(f"Error training the local model: {e}")
        client_socket.close()
        exit(1)

    # Send local weights to the server
    print("Sending local weights to the server...")
    try:
        send_weights_to_server(client_socket, local_weights)
    except Exception as e:
        print("Failed to send weights to server. Exiting.")
        client_socket.close()
        exit(1)

    # Receive global weights from the server (port 9091)
    print("Receiving global weights from the server...")
    try:
        global_weights = receive_weights_from_server(9091)
        # Save global weights to a file
        save_weights_to_file(global_weights, "global_weights_client1.pkl")
    except Exception as e:
        print("Failed to receive global weights. Exiting.")
        client_socket.close()
        exit(1)

    # Update the local model with global weights
    try:
        print("Updating local model with global weights...")
        update_model_weights(global_weights)
        print("Model updated successfully.")
    except Exception as e:
        print(f"Error updating the model with global weights: {e}")

    # Close the connection
    client_socket.close()
    print("Connection closed.")