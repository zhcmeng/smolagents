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
        if getattr(self.tracked_llm_engine, "last_input_token_count", "Not found") != "Not found":
            self.total_input_token_count = 0
            self.total_output_token_count = 0

    def update_metrics(self, step_log):
        step_duration = step_log.step_duration
        self.step_durations.append(step_duration)
        console.print(f"Step {len(self.step_durations)}:")
        console.print(f"- Time taken: {step_duration:.2f} seconds")

        if getattr(self.tracked_llm_engine, "last_input_token_count", None) is not None:
            self.total_input_token_count += self.tracked_llm_engine.last_input_token_count
            self.total_output_token_count += self.tracked_llm_engine.last_output_token_count
            console.print(f"- Input tokens: {self.total_input_token_count:,}")
            console.print(f"- Output tokens: {self.total_output_token_count:,}")


from typing import Optional, Union, List, Any
import httpx
import logging
import os
from langfuse.client import Langfuse, StatefulTraceClient, StatefulSpanClient, StateType


class BaseTracker:
    def __init__(self):
        pass

    @classmethod
    def call(cls, *args, **kwargs):
        pass

class LangfuseTracker(BaseTracker):
    log = logging.getLogger("langfuse")

    def __init__(self, *, public_key: Optional[str] = None, secret_key: Optional[str] = None,
                 host: Optional[str] = None, debug: bool = False, stateful_client: Optional[
                Union[StatefulTraceClient, StatefulSpanClient]
            ] = None, update_stateful_client: bool = False, version: Optional[str] = None,
                 session_id: Optional[str] = None, user_id: Optional[str] = None, trace_name: Optional[str] = None,
                 release: Optional[str] = None, metadata: Optional[Any] = None, tags: Optional[List[str]] = None,
                 threads: Optional[int] = None, flush_at: Optional[int] = None, flush_interval: Optional[int] = None,
                 max_retries: Optional[int] = None, timeout: Optional[int] = None, enabled: Optional[bool] = None,
                 httpx_client: Optional[httpx.Client] = None, sdk_integration: str = "default") -> None:
        super().__init__()
        self.version = version
        self.session_id = session_id
        self.user_id = user_id
        self.trace_name = trace_name
        self.release = release
        self.metadata = metadata
        self.tags = tags

        self.root_span = None
        self.update_stateful_client = update_stateful_client
        self.langfuse = None

        prio_public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        prio_secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        prio_host = host or os.environ.get(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )

        if stateful_client and isinstance(stateful_client, StatefulTraceClient):
            self.trace = stateful_client
            self._task_manager = stateful_client.task_manager
            return

        elif stateful_client and isinstance(stateful_client, StatefulSpanClient):
            self.root_span = stateful_client
            self.trace = StatefulTraceClient(
                stateful_client.client,
                stateful_client.trace_id,
                StateType.TRACE,
                stateful_client.trace_id,
                stateful_client.task_manager,
            )
            self._task_manager = stateful_client.task_manager
            return

        args = {
            "public_key": prio_public_key,
            "secret_key": prio_secret_key,
            "host": prio_host,
            "debug": debug,
        }

        if release is not None:
            args["release"] = release
        if threads is not None:
            args["threads"] = threads
        if flush_at is not None:
            args["flush_at"] = flush_at
        if flush_interval is not None:
            args["flush_interval"] = flush_interval
        if max_retries is not None:
            args["max_retries"] = max_retries
        if timeout is not None:
            args["timeout"] = timeout
        if enabled is not None:
            args["enabled"] = enabled
        if httpx_client is not None:
            args["httpx_client"] = httpx_client
        args["sdk_integration"] = sdk_integration

        self.langfuse = Langfuse(**args)
        self.trace: Optional[StatefulTraceClient] = None
        self._task_manager = self.langfuse.task_manager

    def call(self, i, o, name=None, **kwargs):
        self.langfuse.trace(input=i, output=o, name=name, metadata=kwargs)