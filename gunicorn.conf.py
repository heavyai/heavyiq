import json
import multiprocessing
import threading
from typing import Any

import jwt


def cache_license_edition_background_task(conf_file_path: str):
    """
    Find and set license edition on the shared cache dict.
    """
    from heavyiq.config import get_config, get_heavydb_license_claims
    from heavyiq.utils import SharedDictSingleton

    instance = SharedDictSingleton()  # type: ignore

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
            # 5 (retries) * 8 (timeout) = 40 seconds, so this loop gets excecuted for atmost 40 secs
            retry_count += 1
    else:
        # return None if it can't get the license info after specific retries
        # print("Background task cache_license_edition fails after specific retries.")
        return None

    license_edition = jwt.decode(license_info.claims[0], options={"verify_signature": False})["edition"]

    # print(f"Setting license edition in the shared cache, license_edition: {license_edition}")

    instance.sput(SharedDictSingleton.Keys.HeavyDBLicenseEdition.name, license_edition)

    # print("Background task cache_license_edition completed.")


def run_background_task_in_thread(conf_file_path: str):
    thread = threading.Thread(target=cache_license_edition_background_task, args=(conf_file_path,))
    thread.start()


def print_gunicorn_args(settings: dict):
    """
    Supposed to print gunicorn args.
    Passed args should get higher priority than the defined arguments.
    """
    # print(settings.keys())
    # Selecting important keys
    # important_keys = ["workers", "worker_class", "bind", "preload", "default_proc_name"]
    filtered_args = {key: str(value.get()) for key, value in settings.items()}

    formatted_args = json.dumps(filtered_args, indent=2)
    print("Gunicorn Args (filtered):")
    print(formatted_args)


def on_starting(server: Any):
    """
    This event would be triggered after calling create_app function.
    So this gets called after calling app_initialize function when starting gunicorn with preload option.

    What it does actually?

    1. Print Server Args
    2. Checks for the license edition in background and enables langsmith tracing only for the free edition.
    """

    gunicorn_args = server.cfg.settings
    # print_gunicorn_args(gunicorn_args)

    proc_name = gunicorn_args["default_proc_name"].get()
    conf_file_path = proc_name.split("(")[1].split(")")[0].split("=")[-1].strip("'").strip('"')
    run_background_task_in_thread(conf_file_path)


def post_worker_init(worker: Any):
    import atexit
    from multiprocessing.util import _exit_function

    atexit.unregister(_exit_function)


def on_exit(server: Any):
    from heavyiq.utils import SharedDictSingleton

    shared_instance = SharedDictSingleton._instance  # type: ignore
    if shared_instance and (shared_dict_manager := shared_instance._manager) and shared_dict_manager._state.value == 1:  # type: ignore
        print("Shutdowning shared instance")
        shared_dict_manager.shutdown()
    print("Server exiting...")


cores = multiprocessing.cpu_count()
workers_per_core = 2
workers = 2 * cores  # overrided by passing cli arg -w <num_workers>
worker_class = "uvicorn.workers.UvicornWorker"
bind = "127.0.0.1:8000"
preload_app = True
accesslog = "-"
