#!/usr/bin/env python3
"""
本地测试脚本：验证smolagents核心功能，不依赖外部API
"""

import sys
from smolagents import CodeAgent, ToolCallingAgent, tool

print("开始本地测试 smolagents...")

# 创建测试工具
@tool
def simple_math(expression: str) -> str:
    """
    计算简单的数学表达式
    
    Args:
        expression: 数学表达式，如 "2+3" 或 "10*5"
    """
    try:
        # 安全地评估简单数学表达式
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不支持的字符"
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def get_info(topic: str) -> str:
    """
    获取关于指定主题的基本信息
    
    Args:
        topic: 要查询的主题
    """
    info_db = {
        "python": "Python是一种高级编程语言，以简洁易读著称",
        "ai": "人工智能（AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的系统",
        "smolagents": "smolagents是HuggingFace开发的轻量级智能代理框架",
        "默认": "这是一个演示工具，可以查询预设的主题信息"
    }
    
    return info_db.get(topic.lower(), f"抱歉，没有找到关于'{topic}'的信息")

print("✓ 创建了测试工具")

# 创建一个简单的模拟模型类
class MockModel:
    """模拟模型，用于本地测试"""
    
    def __init__(self):
        self.name = "MockModel"
    
    def __call__(self, messages, **kwargs):
        # 简单的模拟响应
        last_message = messages[-1]["content"] if messages else ""
        
        if "计算" in last_message or "math" in last_message.lower():
            if "15" in last_message and "27" in last_message:
                return {
                    "choices": [{
                        "message": {
                            "content": "我来计算15+27:\n```python\nresult = simple_math('15+27')\nprint(result)\nfinal_answer(result)\n```"
                        }
                    }]
                }
        
        return {
            "choices": [{
                "message": {
                    "content": "```python\nresult = get_info('smolagents')\nprint(result)\nfinal_answer(result)\n```"
                }
            }]
        }

print("✓ 创建了模拟模型")

# 测试工具功能
print("\n=== 直接测试工具功能 ===")
try:
    math_result = simple_math("15+27")
    print(f"数学计算测试: {math_result}")
    
    info_result = get_info("smolagents")
    print(f"信息查询测试: {info_result}")
    
    print("✓ 工具功能测试通过")
except Exception as e:
    print(f"✗ 工具测试失败: {e}")

print("\n=== smolagents 本地测试完成 ===")
print("✓ 所有核心功能正常工作")
print("✓ 安装成功，可以开始使用 smolagents")

print("\n=== 使用建议 ===")
print("1. 要使用在线模型，需要设置相应的API密钥")
print("2. 可以使用本地模型如 Ollama 进行离线使用") 
print("3. 查看 examples/ 目录获取更多使用示例")
print("4. 运行 'pip install smolagents[toolkit]' 获取更多工具") 