import multiprocessing

# Reduce number of workers to minimize memory usage
workers = 1  # Single worker to reduce memory footprint
threads = 2  # Reduce threads to save memory
worker_class = 'gthread'  # Use threads for concurrency

# Timeouts
timeout = 300  # 5 minutes for worker startup
graceful_timeout = 120

# Memory management
max_requests = 50  # Restart workers more frequently to clear memory
max_requests_jitter = 5  # Add randomness to prevent all workers restarting simultaneously

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Prevent workers from starting new requests while shutting down
worker_abort_on_error = True

# Pre-fork settings
preload_app = False  # Don't preload app to reduce initial memory usage

# Bind to address and port
bind = "0.0.0.0:8080"

# Keep alive
keep_alive = 5

# Worker settings
worker_tmp_dir = '/dev/shm'  # Use memory for temp files
