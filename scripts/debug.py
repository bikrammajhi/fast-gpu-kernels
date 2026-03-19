import modal

image = modal.Image.from_registry(
    "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04",
    add_python="3.12",
).add_local_dir("./repos", remote_path="/root/gpulab/repos")

app = modal.App("debug", image=image)

@app.function(gpu="B200")
def check():
    import os
    print("=== /root/gpulab ===")
    for root, dirs, files in os.walk("/root/gpulab"):
        level = root.replace("/root/gpulab", "").count(os.sep)
        if level > 4:
            continue
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:5]:
            print(f"{indent}  {f}")

@app.local_entrypoint()
def main():
    check.remote()
