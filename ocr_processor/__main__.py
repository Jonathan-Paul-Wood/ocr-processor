from waitress import serve

from .app import create_app


if __name__ == "__main__":
    # Exactly one API process; the admission semaphore is process-local.
    serve(create_app(), host="0.0.0.0", port=8090, threads=4,
          max_request_body_size=20 * 1024 * 1024, channel_timeout=240)

