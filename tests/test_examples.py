# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock, skip
from typing import List

from .test_utils import slow, skip, get_launch_command, TempDirTestCase

# DataLoaders built from `test_samples/MRPC` for quick testing
# Should mock `{script_name}.get_dataloaders` via:
# @mock.patch("{script_name}.get_dataloaders", mocked_dataloaders)

EXCLUDE_EXAMPLES = [
    "cross_validation.py",
    "checkpointing.py",
    "gradient_accumulation.py",
    "local_sgd.py",
    "multi_process_metrics.py",
    "memory.py",
    "schedule_free.py",
    "tracking.py",
    "automatic_gradient_accumulation.py",
    "fsdp_with_peak_mem_tracking.py",
    "deepspeed_with_config_support.py",
    "megatron_lm_gpt_pretraining.py",
    "early_stopping.py",
    "ddp_comm_hook.py",
    "profiler.py",
]

import subprocess


class SubprocessCallException(Exception):
    pass


def run_command(command: List[str], return_stdout=False, env=None):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occured while running `command`
    """
    # Cast every path in `command` to a string
    for i, c in enumerate(command):
        if isinstance(c, Path):
            command[i] = str(c)
    if env is None:
        env = os.environ.copy()
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, env=env)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


class ExamplesTests(TempDirTestCase):
    clear_on_setup = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp()
        cls.launch_args = ["python3"]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls._tmpdir)


    def test_oneshot(self):
        testargs = ["examples/oneshot.py"]
        run_command(self.launch_args + testargs)
