#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSeek模型在smolagents中的使用示例

该文件演示了在smolagents项目中集成DeepSeek模型的多种方式
"""

import os
from smolagents import OpenAIServerModel, LiteLLMModel, InferenceClientModel, CodeAgent
from smolagents.default_tools import PythonInterpreterTool, WebSearchTool


def example_openai_server_model():
    """
    方法1: 使用OpenAIServerModel（推荐）
    
    DeepSeek提供完全兼容OpenAI的API接口，这是最直接的集成方式
    """
    print("=== 使用OpenAIServerModel集成DeepSeek ===")
    
    # DeepSeek Chat模型 (DeepSeek-V3-0324)
    model = OpenAIServerModel(
        model_id="deepseek-chat",
        api_base="https://api.deepseek.com",  # 或者使用 "https://api.deepseek.com/v1"
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量获取API密钥
    )
    
    # 创建智能体
    agent = CodeAgent(
        tools=[PythonInterpreterTool(), WebSearchTool()],
        model=model,
        verbose=1
    )
    
    # 测试运行
    result = agent.run("请计算斐波那契数列的前10项")
    print(f"结果: {result}")
    return agent


def example_deepseek_reasoner():
    """
    使用DeepSeek推理模型 (DeepSeek-R1-0528)
    """
    print("=== 使用DeepSeek推理模型 ===")
    
    model = OpenAIServerModel(
        model_id="deepseek-reasoner",  # 推理模型
        api_base="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        # 可以调整推理强度
        reasoning_effort="medium",  # 选项: "low", "medium", "high", "none"
    )
    
    agent = CodeAgent(
        tools=[PythonInterpreterTool()],
        model=model,
        verbose=2
    )
    
    # 测试推理能力
    result = agent.run("解决这个逻辑问题：三个朋友A、B、C，其中只有一个说真话。A说B是骗子，B说C是骗子，C说A和B都是骗子。谁说真话？")
    print(f"推理结果: {result}")
    return agent


def example_litellm_model():
    """
    方法2: 使用LiteLLMModel
    
    通过LiteLLM库访问DeepSeek模型
    """
    print("=== 使用LiteLLMModel集成DeepSeek ===")
    
    model = LiteLLMModel(
        model_id="deepseek/deepseek-chat",  # LiteLLM格式
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.7,
        max_tokens=4096,
    )
    
    agent = CodeAgent(
        tools=[PythonInterpreterTool()],
        model=model
    )
    
    result = agent.run("编写一个Python函数来计算两个数的最大公约数")
    print(f"结果: {result}")
    return agent


def example_inference_client_model():
    """
    方法3: 使用InferenceClientModel
    
    通过HuggingFace推理提供商访问DeepSeek模型
    """
    print("=== 使用InferenceClientModel集成DeepSeek ===")
    
    # 如果DeepSeek在together或其他提供商上可用
    try:
        model = InferenceClientModel(
            model_id="deepseek-ai/DeepSeek-R1",
            provider="together",  # 或其他支持DeepSeek的提供商
            token=os.getenv("HF_TOKEN"),
        )
        
        agent = CodeAgent(
            tools=[WebSearchTool()],
            model=model
        )
        
        result = agent.run("搜索最新的Python机器学习库")
        print(f"结果: {result}")
        return agent
    except Exception as e:
        print(f"InferenceClientModel方式暂不可用: {e}")
        return None


def example_custom_deepseek_class():
    """
    方法4: 创建自定义DeepSeekModel类
    
    这是一个示例，展示如何为DeepSeek创建专门的模型类
    """
    print("=== 自定义DeepSeekModel类示例 ===")
    
    class DeepSeekModel(OpenAIServerModel):
        """
        专门为DeepSeek优化的模型类
        """
        
        def __init__(self, model_type="chat", api_key=None, **kwargs):
            """
            初始化DeepSeek模型
            
            Args:
                model_type: "chat" 或 "reasoner"
                api_key: DeepSeek API密钥
                **kwargs: 其他参数
            """
            model_mapping = {
                "chat": "deepseek-chat",
                "reasoner": "deepseek-reasoner"
            }
            
            super().__init__(
                model_id=model_mapping.get(model_type, "deepseek-chat"),
                api_base="https://api.deepseek.com",
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                **kwargs
            )
            
            self.model_type = model_type
    
    # 使用自定义类
    model = DeepSeekModel(
        model_type="chat",
        temperature=0.7
    )
    
    agent = CodeAgent(
        tools=[PythonInterpreterTool()],
        model=model
    )
    
    result = agent.run("创建一个简单的待办事项管理器")
    print(f"结果: {result}")
    return agent


def example_cli_usage():
    """
    CLI使用示例
    """
    print("=== CLI使用DeepSeek的示例命令 ===")
    
    cli_examples = [
        # 使用OpenAIServerModel方式
        'python -m smolagents.cli "解释量子计算的基本原理" '
        '--model-type OpenAIServerModel '
        '--model-id deepseek-chat '
        '--api-base https://api.deepseek.com '
        '--api-key $DEEPSEEK_API_KEY '
        '--tools web_search python_interpreter',
        
        # 使用LiteLLMModel方式
        'python -m smolagents.cli "编写一个排序算法" '
        '--model-type LiteLLMModel '
        '--model-id deepseek/deepseek-chat '
        '--api-key $DEEPSEEK_API_KEY '
        '--tools python_interpreter',
        
        # 使用推理模型
        'python -m smolagents.cli "解决这个数学难题" '
        '--model-type OpenAIServerModel '
        '--model-id deepseek-reasoner '
        '--api-base https://api.deepseek.com '
        '--api-key $DEEPSEEK_API_KEY'
    ]
    
    for i, cmd in enumerate(cli_examples, 1):
        print(f"\n示例命令 {i}:")
        print(cmd)


def main():
    """
    主函数，演示所有集成方式
    """
    print("DeepSeek模型在smolagents中的集成示例\n")
    
    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("警告: 未设置DEEPSEEK_API_KEY环境变量")
        print("请设置后再运行示例: export DEEPSEEK_API_KEY=your_api_key")
        return
    
    try:
        # 演示各种集成方式
        example_openai_server_model()
        print("\n" + "="*50 + "\n")
        
        example_deepseek_reasoner()
        print("\n" + "="*50 + "\n")
        
        example_litellm_model()
        print("\n" + "="*50 + "\n")
        
        example_inference_client_model()
        print("\n" + "="*50 + "\n")
        
        example_custom_deepseek_class()
        print("\n" + "="*50 + "\n")
        
        example_cli_usage()
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请检查API密钥和网络连接")


if __name__ == "__main__":
    main() 