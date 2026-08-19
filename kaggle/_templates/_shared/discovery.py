"""Finding mounted inputs, and refusing to run on the wrong GPU.

Included verbatim into generated kernels by eda/generate_kernels.py.
Kaggle script kernels are single files, so sharing code means splicing it
at generation time. Editing it here changes every kernel that includes it.
"""

def find_all_markers(pattern: str, max_depth: int = 4) -> list[Path]:
    """Every mounted directory containing a file matching `pattern`.

    The cache is built as four shard kernels and mounted as four separate
    inputs. Finding only the first would silently train on a quarter of the
    data at full apparent success — the worst kind of bug, because the loss
    curve would look fine.
    """
    found = []
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            entries = sorted(directory.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        if any(e.is_file() and e.match(pattern) for e in entries):
            found.append(directory)
        for entry in entries:
            if entry.is_dir() and entry.name not in SKIP_DIRECTORIES:
                frontier.append((entry, depth + 1))
    return found


def find_marker(marker: str, max_depth: int = 4):
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            entries = sorted(directory.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for entry in entries:
            if entry.is_file() and entry.name == marker:
                return directory
        for entry in entries:
            if entry.is_dir() and entry.name not in SKIP_DIRECTORIES:
                frontier.append((entry, depth + 1))
    return None


def report_environment() -> bool:
    """Record the accelerator actually granted.

    The Kaggle CLI does not expose the valid `machine_shape` strings, so which
    GPU a kernel receives has been UNVERIFIED for this project. This prints it,
    which matters because the current PyTorch build ships no Pascal kernels and
    a P100 would fail rather than run slowly.

    Returns True if the accelerator can actually run this build.
    """
    import torch

    print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  no GPU visible; this will be very slow")
        return True
    usable = True
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {name}  compute capability {major}.{minor}")
        if major < 7:
            usable = False
            print("  >>> PRE-VOLTA GPU. The Kaggle PyTorch build ships no Pascal")
            print("  >>> kernels, so every CUDA launch fails with")
            print("  >>> 'no kernel image is available for execution on the device'.")
            print("  >>> Push with --accelerator set to a T4 shape instead.")
    return usable
