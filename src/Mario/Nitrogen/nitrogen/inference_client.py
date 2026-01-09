import time
import pickle

import numpy as np
import zmq

class ModelClient:
    """Client for model inference server."""
    
    def __init__(self, host="localhost", port=5555):
        """
        Initialize client connection.
        
        Args:
            host: Server hostname or IP
            port: Server port
        """
        self.host = host
        self.port = port
        self.timeout_ms = 30000

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)  # Set receive timeout
        
        print(f"Connected to model server at {host}:{port}")
    
    def predict(self, image: np.ndarray) -> dict:
        """
        Send an image and receive predicted actions.
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            List of action dicts, each containing:
                - j_left: [x, y] left joystick position
                - j_right: [x, y] right joystick position  
                - buttons: list of button values
        """
        request = {
            "type": "predict",
            "image": image
        }
        
        self.socket.send(pickle.dumps(request))
        response = pickle.loads(self.socket.recv())
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        return response["pred"]
    
    def reset(self):
        """Reset the server's session (clear buffers)."""
        request = {"type": "reset"}
        
        self.socket.send(pickle.dumps(request))
        response = pickle.loads(self.socket.recv())
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        print("Session reset")

    def info(self) -> dict:
        """Get session info from the server."""
        request = {"type": "info"}
        
        self.socket.send(pickle.dumps(request))
        response = pickle.loads(self.socket.recv())
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        return response["info"]

    def close(self):
        """Close the connection."""
        self.socket.close()
        self.context.term()
        print("Connection closed")
    
    def __enter__(self):
        """Support for context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context."""
        self.close()
