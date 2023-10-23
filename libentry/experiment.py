#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2023-09-06
"""

import io
import os
import random
import shlex
import shutil
import sys
import tarfile
from collections import defaultdict
from datetime import datetime

__all__ = [
    'fix_random_seed',
    'Experiment'
]


def fix_random_seed(
        seed: int = 42,
        for_numpy=True,
        for_torch=True,
        use_deterministic_algorithms=True
):
    os.environ['PYTHONHASHSEED'] = str(seed)
    if use_deterministic_algorithms:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)

    if for_numpy:
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError('numpy not found. pip install numpy')
        np.random.seed(seed)

    if for_torch:
        try:
            import torch
        except ImportError:
            raise RuntimeError('torch not found. pip install torch.')
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        if use_deterministic_algorithms:
            torch.use_deterministic_algorithms(True)


class Experiment(object):
    INSTANCE = None  # type: Experiment

    def __init__(self, experiment_dir, project_dir='.'):
        if Experiment.INSTANCE is not None:
            raise RuntimeError('There is already one experiment instance.')
        Experiment.INSTANCE = self

        self.experiment_dir = experiment_dir
        self.project_dir = project_dir

        if os.path.exists(experiment_dir):
            raise RuntimeError(f'{experiment_dir} already exists.')

        os.makedirs(experiment_dir, exist_ok=True)
        self._backup_project(project_dir, os.path.join(experiment_dir, 'src'))
        self.log('command.txt', ' '.join(map(shlex.quote, sys.argv)))

        self.counter = defaultdict(int)

    def log(self, filename, content, end='\n'):
        with open(os.path.join(self.experiment_dir, filename), 'a') as f:
            f.write(content)
            f.write(end)

    def log_tensor(self, name, tensor):
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError('numpy not found. pip install numpy')
        try:
            import torch
        except ImportError:
            raise RuntimeError('torch not found. pip install torch.')

        tar_path = os.path.join(self.experiment_dir, name + '.tar')
        self.counter[name] += 1
        filename = f'{self.counter[name]}.npy'
        f = io.BytesIO()
        if torch is not None and isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        assert isinstance(tensor, np.ndarray)
        np.save(f, tensor)
        data = f.getvalue()
        self._append_tar(tar_path, filename, data)

    @staticmethod
    def _append_tar(tar_path, filename, data):
        with tarfile.TarFile(tar_path, 'a') as tar:
            info = tarfile.TarInfo(filename)
            info.size = len(data)
            info.mtime = datetime.now().timestamp()
            tar.addfile(info, fileobj=io.BytesIO(data))

    def _backup_project(self, src, dst):
        excludes = self._get_excludes(dst)
        dir_list, file_list = self._get_creation_list(root=src, excludes=excludes)
        os.makedirs(dst, exist_ok=True)
        for path in dir_list:
            os.mkdir(os.path.join(dst, os.path.relpath(path, src)))
        for path in file_list:
            target_path = os.path.join(dst, os.path.relpath(path, src))
            if os.path.getsize(path) > 1024 * 1024 * 10:
                with open(target_path, 'wb'):
                    pass
                continue
            shutil.copy(path, target_path)

    @staticmethod
    def _get_creation_list(root=None, dir_list=None, file_list=None, excludes=None):
        if dir_list is None:
            dir_list = []
        if file_list is None:
            file_list = []

        for name in os.listdir(root):
            if name.startswith('.'):
                continue

            path = os.path.join(root, name) if root is not None else name
            if excludes is not None:
                if any(os.path.samefile(path, exclude) for exclude in excludes):
                    continue

            if os.path.isdir(path):
                dir_list.append(path)
                Experiment._get_creation_list(path, dir_list, file_list, excludes)
            if os.path.isfile(path):
                file_list.append(path)

        return dir_list, file_list

    @staticmethod
    def _get_excludes(path):
        path = os.path.abspath(path)
        excludes = []
        while path and path != '/':
            if os.path.exists(path):
                excludes.append(path)
            path = os.path.dirname(path)
        return excludes
