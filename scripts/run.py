import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpulab.modal_app import run, app


@app.local_entrypoint()
def main(task: str, params: str = "{}", flags: str = ""):
    import json
    extra_flags = flags.split() if flags else []
    run.remote(task, json.loads(params), extra_flags)
