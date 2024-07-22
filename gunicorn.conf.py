from __future__ import print_function

import json
import multiprocessing
import os
import subprocess
import threading
import time

try:
    import jwt
except ImportError:
    # Ensure jwt is installed or handled appropriately
    pass

chromadb_process, DB_PATH, PORT = None, None, None


def start_chromadb_server_process():
    """
    Helps to start ChromaDB server process.
    """
    global chromadb_process, DB_PATH, PORT

    if "CHROMADB_STARTED" not in os.environ:
        assert DB_PATH and PORT
        # Open log file
        log_file = open("chromadb.log", "a")
        # Start the ChromaDB server and redirect stdout and stderr to the log file
        chromadb_process = subprocess.Popen(
            ["chroma", "run", "--path", DB_PATH, "--port", f"{PORT}"], stdout=log_file, stderr=log_file
        )
        os.environ["CHROMADB_STARTED"] = "1"
        print(f"Started chromadb server...\nArgs:\n--path {DB_PATH}\n--port {PORT}\nSee logs at {log_file.name}")
        return True

    return False


def monitor_chromadb():
    """
    Helps to monitor the server and does automatic restart on failure.
    """
    print("Started monitoring chromaDB server process...")
    while True:
        if chromadb_process and chromadb_process.poll() is not None:
            print("ChromaDB server exited unexpectedly. Restarting...")
            os.environ.pop("CHROMADB_STARTED")
            start_chromadb_server_process()
        time.sleep(5)


def check_and_initiate_chromadb_thread(conf_file_path: str):
    """
    Parser the configuration file and optionally initiate the chromadb server.
    """
    from heavyiq.config import get_config
    from heavyiq.utils import get_host_and_port

    global DB_PATH, PORT

    if conf_file_path:
        config = get_config(conf_file_path)
    else:
        config = get_config()

    server_base = config.rag_chromadb_server_base
    if not server_base:
        print("Failed to start chromadb server, no chromadb server base found on config.")
        return None

    _, PORT = get_host_and_port(server_base)
    DB_PATH = config.rag_chromadb_persist_dir

    is_started = start_chromadb_server_process()
    if not is_started:
        print("Failed to start chromadb server.")
    # run background chromadb monitor thread
    chromadb_monitor_thread()


def cache_license_edition_background_task(conf_file_path: str):
    """
    Find and set license edition on the shared cache dict.
    """
    from heavyiq.config import get_config, get_heavydb_license_claims
    from heavyiq.utils import SharedDictSingleton

    instance = SharedDictSingleton()  # Removed type hint

    if conf_file_path:
        config = get_config(conf_file_path)
    else:
        config = get_config()

    max_retries, retry_count = 5, 0
    while retry_count < max_retries:
        try:
            license_info = get_heavydb_license_claims(config)
            break
        except ValueError:
            # ValueError regarding timeout should be raised after 8 seconds
            # 5 (retries) * 8 (timeout) = 40 seconds, so this loop gets executed for at most 40 secs
            retry_count += 1
    else:
        # return None if it can't get the license info after specific retries
        # print("Background task cache_license_edition fails after specific retries.")
        return None

    license_edition = jwt.decode(license_info.claims[0], options={"verify_signature": False})["edition"]

    # print("Setting license edition in the shared cache, license_edition: %s" % license_edition)

    instance.sput(SharedDictSingleton.Keys.HeavyDBLicenseEdition.name, license_edition)

    # print("Background task cache_license_edition completed.")


def run_background_task_in_thread(conf_file_path: str):
    thread = threading.Thread(target=cache_license_edition_background_task, args=(conf_file_path,))
    thread.start()


def run_background_chromadb_initiate_task_in_thread(conf_file_path: str):
    thread = threading.Thread(target=check_and_initiate_chromadb_thread, args=(conf_file_path,))
    thread.start()


def chromadb_monitor_thread():
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_chromadb, daemon=True)
    monitor_thread.start()


def print_gunicorn_args(settings):
    """
    Supposed to print gunicorn args.
    Passed args should get higher priority than the defined arguments.
    """
    filtered_args = dict((key, str(value.get())) for key, value in settings.items())  # Compatible with both versions

    formatted_args = json.dumps(filtered_args, indent=2)
    # print("Gunicorn Args (filtered):")
    # print(formatted_args)


def on_starting(server):
    """
    This event would be triggered after calling create_app function.
    So this gets called after calling app_initialize function when starting gunicorn with preload option.

    What it does actually?

    1. Print Server Args
    2. Checks for the license edition in background and enables langsmith tracing only for the free edition.
    """

    gunicorn_args = server.cfg.settings
    print_gunicorn_args(gunicorn_args)

    proc_name = gunicorn_args["default_proc_name"].get()
    conf_file_path = proc_name.split("(")[1].split(")")[0].split("=")[-1].strip("'").strip('"')
    # check_and_initiate_chromadb_thread(conf_file_path)
    check_and_initiate_chromadb_thread(conf_file_path)
    run_background_task_in_thread(conf_file_path)


def post_worker_init(worker):
    import atexit
    from multiprocessing.util import _exit_function

    atexit.unregister(_exit_function)


def on_exit(server):
    from heavyiq.utils import SharedDictSingleton

    os.environ.pop("CHROMADB_STARTED", None)

    shared_instance = SharedDictSingleton._instance  # Removed type hint
    if shared_instance and hasattr(shared_instance, "_manager") and shared_instance._manager._state.value == 1:
        # print("Shutting down shared instance")
        shared_instance._manager.shutdown()
    # print("Server exiting...")
    # exit chromadb server
    global chromadb_process
    if chromadb_process:
        chromadb_process.terminate()
        chromadb_process.wait()


cores = multiprocessing.cpu_count()
workers_per_core = 2
workers = 2 * cores  # overridden by passing CLI arg -w <num_workers>
worker_class = "uvicorn.workers.UvicornWorker"
bind = "127.0.0.1:8000"
preload_app = True
accesslog = "-"
