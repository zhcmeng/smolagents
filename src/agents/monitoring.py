#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from .utils import console


class Monitor:
    def __init__(self, tracked_llm_engine):
        self.step_durations = []
        self.tracked_llm_engine = tracked_llm_engine
        if (
            getattr(self.tracked_llm_engine, "last_input_token_count", "Not found")
            != "Not found"
        ):
            self.total_input_token_count = 0
            self.total_output_token_count = 0

    def update_metrics(self, step_log):
        step_duration = step_log.duration
        self.step_durations.append(step_duration)
        console.print(f"Step {len(self.step_durations)}:")
        console.print(f"- Time taken: {step_duration:.2f} seconds")

        if getattr(self.tracked_llm_engine, "last_input_token_count", None) is not None:
            self.total_input_token_count += (
                self.tracked_llm_engine.last_input_token_count
            )
            self.total_output_token_count += (
                self.tracked_llm_engine.last_output_token_count
            )
            console.print(f"- Input tokens: {self.total_input_token_count:,}")
            console.print(f"- Output tokens: {self.total_output_token_count:,}")

__all__ = ["Monitor"]