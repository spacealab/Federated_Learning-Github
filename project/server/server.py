import socket
import pickle
import threading
import os

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

def load_weights_from_file(filename):
    """
    Load weights from a file.
    """
    try:
        with open(filename, "rb") as f:
            weights = pickle.load(f)
        print(f"Weights loaded from {filename}.")
        return weights
    except Exception as e:
        print(f"Error loading weights from file: {e}")
        return None

def aggregate_weights(client_weights):
    """
    Average the weights from clients to update global weights.
    """
    print("Aggregating weights from all clients...")
    total_weights = {}
    for client in client_weights:
        for key, value in client.items():
            if key not in total_weights:
                total_weights[key] = value
            else:
                total_weights[key] += value
    return {key: value / len(client_weights) for key, value in total_weights.items()}

def handle_client(conn, addr, client_weights, num_clients, lock):
    """
    Handle communication with a single client.
    """
    try:
        print(f"Connection established with {addr}")

        # Receive data from client
        data = b""
        while True:
            packet = conn.recv(4096)  # Receive in chunks
            if not packet:
                break
            data += packet

        if data:
            received_weights = pickle.loads(data)
            print(f"Received weights from {addr}")

            # Save received weights to a file
            local_weights_filename = f"local_weights_{addr[0]}_{addr[1]}.pkl"
            save_weights_to_file(received_weights, local_weights_filename)

            with lock:
                client_weights.append(received_weights)  # Store weights

                # Check if all clients have sent their weights
                if len(client_weights) >= num_clients:
                    print("All clients have sent their weights. Aggregating...")
                    global_weights = aggregate_weights(client_weights)

                    # Save global weights to a file
                    global_weights_filename = "global_weights.pkl"
                    save_weights_to_file(global_weights, global_weights_filename)

                    # Send global weights back to all clients through a new port
                    print("Sending global weights to all clients through port 9091...")
                    send_global_weights_to_clients(global_weights)

                    # Reset the client weights after aggregation
                    client_weights.clear()

    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        conn.close()
        print(f"Connection with {addr} closed.")

def send_global_weights_to_clients(global_weights):
    """
    Send global weights to all clients through a new port.
    """
    try:
        # Create a new socket for sending global weights
        send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        send_socket.bind(("127.0.0.1", 9091))  # Use a new port (9091)
        send_socket.listen(5)

        print("Waiting for clients to connect to port 9091...")
        while True:
            client_conn, client_addr = send_socket.accept()
            print(f"Client {client_addr} connected to receive global weights.")
            try:
                client_conn.sendall(pickle.dumps(global_weights))
                print(f"Sent global weights to {client_addr}.")
            except Exception as e:
                print(f"Error sending global weights to {client_addr}: {e}")
            finally:
                client_conn.close()
                print(f"Connection with {client_addr} closed.")
                break  # Exit after sending to one client
    except Exception as e:
        print(f"Error in send_global_weights_to_clients: {e}")
    finally:
        send_socket.close()

def run_server():
    """
    Run the server to receive and send weights.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 9090))  # Server address and port
    server_socket.listen(5)

    print("Server is waiting for connections...")

    client_weights = []
    num_clients = 2  # Define how many clients are expected (2 clients)
    lock = threading.Lock()  # Lock for thread-safe operations

    try:
        while True:
            conn, addr = server_socket.accept()
            print(f"Accepted connection from {addr}")
            client_thread = threading.Thread(target=handle_client, args=(conn, addr, client_weights, num_clients, lock))
            client_thread.start()
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        server_socket.close()

if __name__ == "__main__":
    run_server()