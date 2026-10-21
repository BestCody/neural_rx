import argparse
import os
import subprocess

DEFAULT_DETECTOR_BATCH_TBS = 8
DEFAULT_DESCRIPTOR_CHUNK_TBS = 32
DEFAULT_RENDER_BATCH_TBS = 16


def requested_updates(steps):
    if steps is None:
        raise ValueError("Specify --steps; training has no implicit pilot duration")
    if type(steps) is not int or steps < 0:
        raise ValueError("The total update count must be a nonnegative integer")
    return steps


def data_gpu(value):
    if value == "auto":
        return value
    if value == "none":
        return None
    try:
        index = int(value)
        if index >= 0:
            return index
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use auto, none, or a nonnegative GPU index")


def add_preparation_arguments(parser):
    parser.add_argument("--render-batch-tbs", type=int, default=DEFAULT_RENDER_BATCH_TBS,
                        help="TBs constructed together without changing per-TB seeds")
    parser.add_argument("--data-gpu", type=data_gpu, default="auto",
                        help="Separate preparation GPU: auto (default), none, or an index")
    parser.add_argument("--prefetch-examples", type=int, default=0,
                        help="Waiting/in-progress slots per stage; default 0 uses one effective update")
    parser.add_argument("--channel-workers", type=int,
                        help="Default: one when sharing with a learner, two with a dedicated data GPU")
    parser.add_argument("--prefetch-memory-mib", type=int, default=4096,
                        help="Total retained array budget across producer queues (default: 4096 MiB)")
    parser.add_argument("--replica-gpu", type=data_gpu, default="auto",
                        help="Additional synchronous learner: auto, none, or GPU index")
    parser.add_argument("--replica-memory-mib", type=int, default=0,
                        help="Additional learner allocator budget; 0 divides a shared GPU with preparation")
    parser.add_argument("--preparation-gpu-memory-mib", type=int, default=0,
                        help="Total producer allocator budget on a shared GPU; 0 selects it automatically")


def add_execution_arguments(parser):
    parser.add_argument("--detector-batch-tbs", type=int, default=DEFAULT_DETECTOR_BATCH_TBS,
                        help="TB inputs per gradient batch, each with persistent and reset supervision")
    parser.add_argument("--descriptor-chunk-tbs", type=int, default=DEFAULT_DESCRIPTOR_CHUNK_TBS)


def configure_gpu(gpu, memory_mib=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    for device in tf.config.list_physical_devices("GPU"):
        if memory_mib:
            tf.config.set_logical_device_configuration(device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_mib))])
        else:
            tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.enable_op_determinism()
    return tf


def visible_gpu_indices():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5)
        devices = {int(index.strip()): uuid.strip()
                   for index, uuid in (line.split(",", 1) for line in result.stdout.splitlines())}
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return sorted(devices)
    selected = []
    for token in visible.split(","):
        token = token.strip()
        matches = [index for index, uuid in devices.items()
                   if token == str(index) or (token.startswith("GPU-") and uuid.startswith(token))]
        if len(matches) != 1 or matches[0] in selected:
            break
        selected.append(matches[0])
    return selected


def resolve_preparation(args, effective_examples):
    if args.render_batch_tbs < 1:
        raise ValueError("The renderer batch capacity must be positive")
    if args.prefetch_examples < 0 or effective_examples < 1:
        raise ValueError("Invalid preparation capacity")
    if args.prefetch_memory_mib < 1:
        raise ValueError("The prefetch memory budget must be positive")
    capacity = args.prefetch_examples or effective_examples
    if args.data_gpu == "auto":
        args.data_gpu = next((gpu for gpu in visible_gpu_indices() if gpu != args.gpu), None)
    if args.data_gpu == args.gpu:
        raise ValueError("Background preparation requires a separate GPU")
    replica = args.replica_gpu
    if replica == "auto":
        replica = next((gpu for gpu in visible_gpu_indices() if gpu != args.gpu), None)
    if replica == args.gpu:
        raise ValueError("The additional learner requires a different GPU")
    args.replica_gpu = replica
    if min(args.replica_memory_mib, args.preparation_gpu_memory_mib) < 0:
        raise ValueError("GPU memory budgets cannot be negative")
    shared = replica is not None and replica == args.data_gpu
    if shared:
        result = subprocess.run(["nvidia-smi", f"--id={replica}", "--query-gpu=memory.total",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        total = int(result.stdout.strip())
        budget = total-max(1024, total//20)
        if not args.replica_memory_mib:
            args.replica_memory_mib = budget//2
        if not args.preparation_gpu_memory_mib:
            args.preparation_gpu_memory_mib = budget-args.replica_memory_mib
        if min(args.replica_memory_mib, args.preparation_gpu_memory_mib) <= 0 or (
                args.replica_memory_mib+args.preparation_gpu_memory_mib > budget):
            raise ValueError("Shared learner and producer budgets exceed GPU memory with context headroom")
    if args.channel_workers is None:
        args.channel_workers = min(1 if shared else 2, capacity) if args.data_gpu is not None else 1
    if args.channel_workers < 1:
        raise ValueError("Invalid preparation concurrency")
    if args.channel_workers > 1 and args.data_gpu is None:
        raise ValueError("Parallel channels require a separate data GPU")
    if args.data_gpu is not None and args.channel_workers > capacity:
        raise ValueError("Each channel worker requires at least one channel slot")


def preparation_arguments(args):
    return ["--render-batch-tbs", str(args.render_batch_tbs),
            "--data-gpu", "none" if args.data_gpu is None else str(args.data_gpu),
            "--prefetch-examples", str(args.prefetch_examples),
            "--channel-workers", str(args.channel_workers),
            "--prefetch-memory-mib", str(args.prefetch_memory_mib),
            "--replica-gpu", "none" if args.replica_gpu is None else str(args.replica_gpu),
            "--replica-memory-mib", str(args.replica_memory_mib),
            "--preparation-gpu-memory-mib", str(args.preparation_gpu_memory_mib)]
