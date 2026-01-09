import zmq
import argparse
import pickle

from nitrogen.inference_session import InferenceSession

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference server")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("--port", type=int, default=5555, help="Port to serve on")
    parser.add_argument("--old-layout", action="store_true", help="Use old layout")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--ctx", type=int, default=1, help="Context length")
    args = parser.parse_args()

    session = InferenceSession.from_ckpt(args.ckpt, old_layout=args.old_layout, cfg_scale=args.cfg, context_length=args.ctx)

    # Setup ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    # Create poller
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print(f"\n{'='*60}")
    print(f"Server running on port {args.port}")
    print(f"Waiting for requests...")
    print(f"{'='*60}\n")

    try:
        while True:
            # Poll with 100ms timeout to allow interrupt handling
            events = dict(poller.poll(timeout=100))
            if socket in events and events[socket] == zmq.POLLIN:
                # Receive request only when data is available
                request = socket.recv()
                request = pickle.loads(request)
                if request["type"] == "reset":
                    session.reset()
                    response = {"status": "ok"}
                    print("Session reset")
                elif request["type"] == "info":
                    info = session.info()
                    response = {"status": "ok", "info": info}
                    print("Sent session info")
                elif request["type"] == "predict":
                    raw_image = request["image"]
                    result = session.predict(raw_image)
                    response = {
                        "status": "ok",
                        "pred": result
                    }
                else:
                    response = {"status": "error", "message": f"Unknown request type: {request['type']}"}
                # Send response
                socket.send(pickle.dumps(response))
    except KeyboardInterrupt:
        print("\nShutting down server...")
        exit(0)
    finally:
        socket.close()
        context.term()
