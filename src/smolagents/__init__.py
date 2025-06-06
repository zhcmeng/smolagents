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

"""
smolagents - 轻量级智能代理框架

这是一个功能强大且易于使用的智能代理框架，支持多种 LLM 模型和工具集成。

主要特性:
- 🤖 支持多种代理类型（CodeAgent、ToolCallingAgent）
- 🛠️ 丰富的工具生态系统和自定义工具支持
- 🔄 基于 ReAct 框架的推理-行动循环
- 🌐 多种模型提供商支持（OpenAI、HuggingFace、本地模型等）
- 📊 实时监控和日志记录
- 🎯 流式输出和批量处理
- 🔧 灵活的执行环境（本地、Docker、E2B）

快速开始:
```python
from smolagents import CodeAgent, InferenceClientModel

# 创建模型和代理
model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[], model=model)

# 运行任务
result = agent.run("计算 2 的 10 次方")
print(result)
```

作者: HuggingFace 团队
版本: 1.18.0.dev0
许可证: Apache 2.0
"""

__version__ = "1.18.0.dev0"

from .agent_types import *  # noqa: I001
from .agents import *  # Above noqa avoids a circular dependency due to cli.py
from .default_tools import *
from .gradio_ui import *
from .local_python_executor import *
from .mcp_client import *
from .memory import *
from .models import *
from .monitoring import *
from .remote_executors import *
from .tools import *
from .utils import *
from .cli import *
