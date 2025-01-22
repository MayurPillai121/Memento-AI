import multiprocessing

# Reduce number of workers to minimize memory usage
workers = 1  # Single worker to reduce memory footprint
threads = 4  # Maintain some concurrency with threads
worker_class = 'gthread'  # Use threads for concurrency

# Timeouts
timeout = 600  # 10 minutes
graceful_timeout = 120

# Memory management
max_requests = 100  # Restart workers periodically to clear memory
max_requests_jitter = 10  # Add randomness to prevent all workers restarting simultaneously

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

# Worker connections
worker_connections = 1000
