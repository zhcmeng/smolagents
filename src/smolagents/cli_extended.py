#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
扩展的CLI实现，增加对DeepSeek的专门支持
"""

import argparse
import os

from dotenv import load_dotenv

from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, Model, OpenAIServerModel, Tool, TransformersModel
from smolagents.default_tools import TOOL_MAPPING


leopard_prompt = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a CodeAgent with all specified parameters")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=leopard_prompt,
        help="The prompt to run with the agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="InferenceClientModel",
        help="The model type to use (e.g., InferenceClientModel, OpenAIServerModel, LiteLLMModel, TransformersModel, DeepSeekModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--imports",
        nargs="*",  # accepts zero or more arguments
        default=[],
        help="Space-separated list of imports to authorize (e.g., 'numpy pandas')",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=["web_search"],
        help="Space-separated list of tools that the agent can use (e.g., 'tool1 tool2 tool3')",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="The verbosity level, as an int in [0, 1, 2].",
    )
    group = parser.add_argument_group("api options", "Options for API-based model types")
    group.add_argument(
        "--provider",
        type=str,
        default=None,
        help="The inference provider to use for the model",
    )
    group.add_argument(
        "--api-base",
        type=str,
        help="The base URL for the model",
    )
    group.add_argument(
        "--api-key",
        type=str,
        help="The API key for the model",
    )
    
    # DeepSeek专用选项
    deepseek_group = parser.add_argument_group("deepseek options", "Options specific to DeepSeek models")
    deepseek_group.add_argument(
        "--deepseek-model-type",
        type=str,
        choices=["chat", "reasoner"],
        default="chat",
        help="DeepSeek model type: 'chat' for DeepSeek-V3 or 'reasoner' for DeepSeek-R1",
    )
    deepseek_group.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high", "none"],
        default="medium",
        help="Reasoning effort level for DeepSeek reasoner model",
    )
    
    return parser.parse_args()


def create_deepseek_model(
    model_type: str = "chat",
    api_key: str | None = None,
    reasoning_effort: str = "medium",
    **kwargs
) -> OpenAIServerModel:
    """
    创建DeepSeek模型的便捷函数
    
    Args:
        model_type: "chat" 或 "reasoner"
        api_key: DeepSeek API密钥
        reasoning_effort: 推理强度（仅对reasoner模型有效）
        **kwargs: 其他参数
    
    Returns:
        配置好的OpenAIServerModel实例
    """
    model_mapping = {
        "chat": "deepseek-chat",
        "reasoner": "deepseek-reasoner"
    }
    
    model_kwargs = {
        "model_id": model_mapping.get(model_type, "deepseek-chat"),
        "api_base": "https://api.deepseek.com",
        "api_key": api_key or os.getenv("DEEPSEEK_API_KEY"),
        **kwargs
    }
    
    # 为推理模型添加推理强度参数
    if model_type == "reasoner":
        model_kwargs["reasoning_effort"] = reasoning_effort
    
    return OpenAIServerModel(**model_kwargs)


def load_model(
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    # DeepSeek专用参数
    deepseek_model_type: str = "chat",
    reasoning_effort: str = "medium",
) -> Model:
    """
    扩展的模型加载函数，增加DeepSeek支持
    """
    if model_type == "DeepSeekModel":
        # 专门的DeepSeek模型创建
        return create_deepseek_model(
            model_type=deepseek_model_type,
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            reasoning_effort=reasoning_effort,
        )
    elif model_type == "OpenAIServerModel":
        # 检查是否是DeepSeek模型ID
        if model_id in ["deepseek-chat", "deepseek-reasoner"]:
            model_kwargs = {
                "model_id": model_id,
                "api_base": api_base or "https://api.deepseek.com",
                "api_key": api_key or os.getenv("DEEPSEEK_API_KEY"),
            }
            if model_id == "deepseek-reasoner":
                model_kwargs["reasoning_effort"] = reasoning_effort
            return OpenAIServerModel(**model_kwargs)
        else:
            # 原有的OpenAI服务器模型逻辑
            return OpenAIServerModel(
                api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
                api_base=api_base or "https://api.fireworks.ai/inference/v1",
                model_id=model_id,
            )
    elif model_type == "LiteLLMModel":
        # 检查是否是DeepSeek模型
        if model_id.startswith("deepseek/") or model_id in ["deepseek-chat", "deepseek-reasoner"]:
            # 为DeepSeek调整LiteLLM格式
            litellm_model_id = model_id if model_id.startswith("deepseek/") else f"deepseek/{model_id}"
            return LiteLLMModel(
                model_id=litellm_model_id,
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                api_base=api_base,
            )
        else:
            return LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                api_base=api_base,
            )
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto")
    elif model_type == "InferenceClientModel":
        return InferenceClientModel(
            model_id=model_id,
            token=api_key or os.getenv("HF_TOKEN"),
            provider=provider,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_smolagent(
    prompt: str,
    tools: list[str],
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    imports: list[str] | None = None,
    provider: str | None = None,
    # DeepSeek专用参数
    deepseek_model_type: str = "chat",
    reasoning_effort: str = "medium",
) -> None:
    load_dotenv()

    model = load_model(
        model_type, 
        model_id, 
        api_base=api_base, 
        api_key=api_key, 
        provider=provider,
        deepseek_model_type=deepseek_model_type,
        reasoning_effort=reasoning_effort,
    )

    available_tools = []
    for tool_name in tools:
        if "/" in tool_name:
            available_tools.append(Tool.from_space(tool_name))
        else:
            if tool_name in TOOL_MAPPING:
                available_tools.append(TOOL_MAPPING[tool_name]())
            else:
                raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

    print(f"Running agent with these tools: {tools}")
    print(f"Using model: {model_type} ({model_id if model_type != 'DeepSeekModel' else f'deepseek-{deepseek_model_type}'})")
    
    agent = CodeAgent(tools=available_tools, model=model, additional_authorized_imports=imports)

    result = agent.run(prompt)
    print(f"\n最终结果: {result}")


def print_deepseek_examples():
    """
    打印DeepSeek使用示例
    """
    print("\n=== DeepSeek使用示例 ===")
    
    examples = [
        "# 使用DeepSeek Chat模型",
        "python cli_extended.py '编写一个Python爬虫' --model-type DeepSeekModel --deepseek-model-type chat",
        
        "\n# 使用DeepSeek推理模型",
        "python cli_extended.py '解决数学难题' --model-type DeepSeekModel --deepseek-model-type reasoner --reasoning-effort high",
        
        "\n# 通过OpenAIServerModel使用DeepSeek",
        "python cli_extended.py '分析数据' --model-type OpenAIServerModel --model-id deepseek-chat --api-key $DEEPSEEK_API_KEY",
        
        "\n# 通过LiteLLM使用DeepSeek",
        "python cli_extended.py '代码重构' --model-type LiteLLMModel --model-id deepseek/deepseek-chat --api-key $DEEPSEEK_API_KEY",
    ]
    
    for example in examples:
        print(example)


def main() -> None:
    args = parse_arguments()
    
    # 检查DeepSeek相关的环境设置
    if args.model_type == "DeepSeekModel" or args.model_id in ["deepseek-chat", "deepseek-reasoner"]:
        if not args.api_key and not os.getenv("DEEPSEEK_API_KEY"):
            print("警告: 使用DeepSeek模型需要设置API密钥")
            print("请设置环境变量: export DEEPSEEK_API_KEY=your_api_key")
            print("或使用 --api-key 参数")
            print_deepseek_examples()
            return
    
    try:
        run_smolagent(
            args.prompt,
            args.tools,
            args.model_type,
            args.model_id,
            provider=args.provider,
            api_base=args.api_base,
            api_key=args.api_key,
            imports=args.imports,
            deepseek_model_type=args.deepseek_model_type,
            reasoning_effort=args.reasoning_effort,
        )
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        if "deepseek" in str(e).lower() or "api" in str(e).lower():
            print("\n可能的解决方案:")
            print("1. 检查DEEPSEEK_API_KEY是否正确设置")
            print("2. 确认网络连接正常")
            print("3. 验证API密钥是否有效")
            print_deepseek_examples()


if __name__ == "__main__":
    main() 