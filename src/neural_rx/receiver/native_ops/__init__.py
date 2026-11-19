from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from functools import lru_cache
from threading import RLock

import tensorflow as tf


_lock = RLock()


def load():
    with _lock:
        return _load()


@lru_cache(maxsize=1)
def _load():
    directory = Path(__file__).resolve().parent
    sources = sorted(directory.glob('*.cc'))
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    nvcc = shutil.which('nvcc') or str(cuda/'bin/nvcc')
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        raise RuntimeError('The pilot memory requires a CUDA GPU')
    capabilities = sorted({tf.config.experimental.get_device_details(d)['compute_capability'] for d in devices})
    flags, links = tf.sysconfig.get_compile_flags(), tf.sysconfig.get_link_flags()
    libraries = []
    compiler = subprocess.check_output([nvcc, '--version'])
    source_bytes = [p.read_bytes() for p in sources]
    key = hashlib.sha256(b''.join(source_bytes)+Path(__file__).read_bytes()+compiler+
                         json.dumps([tf.__version__, flags, links, libraries, capabilities]).encode()).hexdigest()
    cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))/'neural_rx_native'/key
    cache.mkdir(parents=True, exist_ok=True)
    library = cache/'pilot_memory.so'
    with (cache/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temp:
                temp = Path(temp)
                copies = [temp/p.name for p in sources]
                for path, content in zip(copies, source_bytes, strict=True):
                    path.write_bytes(content)
                architectures = [f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'
                                 for major, minor in capabilities]
                gpu_sources = [p for p in copies if p.name.endswith('.cu.cc')]
                commands = [[nvcc, '-std=c++17', '-O3', '--expt-relaxed-constexpr', '-x', 'cu', '-c', str(p),
                    '-o', str(p.with_suffix('.o')), '-Xcompiler', '-fPIC', '-DGOOGLE_CUDA=1', *flags, *architectures]
                    for p in gpu_sources]
                commands.append(['g++', '-std=c++17', '-O3', '-shared', '-fPIC', '-DGOOGLE_CUDA=1',
                    *[str(p) for p in copies if p not in gpu_sources], *[str(p.with_suffix('.o')) for p in gpu_sources],
                    '-o', str(temp/library.name), '-I'+str(cuda/'include'), '-L'+str(cuda/'lib64'), '-lcudart',
                    *libraries, *flags, *links])
                with (cache/'build.log').open('w') as log:
                    for command in commands:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode:
                            raise RuntimeError(f'Native operation build failed; see {cache / "build.log"}')
                os.replace(temp/library.name, library)
    return tf.load_op_library(str(library))
