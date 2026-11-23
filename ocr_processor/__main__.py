from waitress import serve
import signal

from .app import create_app


if __name__ == "__main__":
    # Exactly one API process; the admission semaphore is process-local.
    app = create_app()
    debug = app.extensions["traffic_debug"]
    if debug:
        def terminate(signum, frame):
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, terminate)
    try:
        serve(app, host="0.0.0.0", port=8081, threads=4,
              max_request_body_size=20 * 1024 * 1024, channel_timeout=240)
    finally:
        if debug:
            debug.close(timeout=2)

