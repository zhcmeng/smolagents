#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用中文提示模板的代理示例
"""

import yaml
from smolagents import CodeAgent, OpenAIModel
from smolagents.default_tools import PythonInterpreterTool


def create_chinese_agent():
    """创建使用中文的代理"""
    
    # 方法1: 加载自定义中文模板文件
    with open('chinese_prompts_example.yaml', 'r', encoding='utf-8') as f:
        chinese_prompts = yaml.safe_load(f)
    
    # 初始化模型
    model = OpenAIModel("gpt-4")
    
    # 创建带有中文提示模板的代理
    agent = CodeAgent(
        tools=[PythonInterpreterTool()],
        model=model,
        prompt_templates=chinese_prompts,
        verbosity_level=1  # 显示详细日志
    )
    
    return agent


def create_chinese_agent_inline():
    """通过内联修改创建中文代理"""
    
    # 方法2: 直接在代码中修改提示模板
    model = OpenAIModel("gpt-4")
    
    # 创建默认代理
    agent = CodeAgent(
        tools=[PythonInterpreterTool()],
        model=model
    )
    
    # 修改系统提示词
    original_prompt = agent.prompt_templates["system_prompt"]
    chinese_instruction = """请严格使用中文进行所有的思考、分析和回复。
在每个'Thought:'部分，必须使用中文进行思考和解释。

"""
    
    # 在原有提示词前加上中文指令
    agent.prompt_templates["system_prompt"] = chinese_instruction + original_prompt
    
    # 修改规则部分（在原有规则中添加中文要求）
    agent.prompt_templates["system_prompt"] += """

重要提醒：所有思考过程必须用中文表达！"""
    
    return agent


def create_chinese_agent_simple():
    """最简单的方法：在系统提示词开头添加中文指令"""
    
    model = OpenAIModel("gpt-4")
    agent = CodeAgent(
        tools=[PythonInterpreterTool()],
        model=model
    )
    
    # 最简单的修改：在系统提示词开头添加中文要求
    agent.prompt_templates["system_prompt"] = """你必须用中文进行所有思考和分析。
在'Thought:'部分，请用中文解释你的推理过程。

""" + agent.prompt_templates["system_prompt"]
    
    return agent


def test_chinese_agent():
    """测试中文代理"""
    
    # 创建中文代理
    agent = create_chinese_agent_simple()
    
    # 测试任务
    task = "计算 15 * 23 + 67 的结果，并解释计算过程"
    
    print("=== 使用中文代理执行任务 ===")
    print(f"任务: {task}")
    print("=" * 50)
    
    # 执行任务
    result = agent.run(task)
    print(f"结果: {result}")


if __name__ == "__main__":
    # 创建不同类型的中文代理示例
    
    print("方法1: 使用自定义YAML模板")
    try:
        agent1 = create_chinese_agent()
        print("✓ 中文代理创建成功")
    except FileNotFoundError:
        print("✗ 需要先创建 chinese_prompts_example.yaml 文件")
    
    print("\n方法2: 内联修改提示模板")
    agent2 = create_chinese_agent_inline()
    print("✓ 中文代理创建成功")
    
    print("\n方法3: 简单添加中文指令")
    agent3 = create_chinese_agent_simple()
    print("✓ 中文代理创建成功")
    
    # 运行测试（需要OpenAI API密钥）
    # test_chinese_agent() 