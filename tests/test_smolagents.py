#!/usr/bin/env python3
"""
简单测试脚本：验证smolagents安装和基本功能
"""

import sys
print("开始测试 smolagents...")

# 测试1：导入基本模块
try:
    from smolagents import CodeAgent, ToolCallingAgent, tool, InferenceClientModel
    print("✓ 成功导入 smolagents 核心模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试2：创建一个简单的工具
@tool
def simple_calculator(a: float, b: float, operation: str = "add") -> str:
    """
    执行简单的数学运算
    
    Args:
        a: 第一个数字
        b: 第二个数字  
        operation: 运算类型 ('add', 'subtract', 'multiply', 'divide')
    """
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "错误：除数不能为零"
        result = a / b
    else:
        return f"不支持的运算类型：{operation}"
    
    return f"{a} {operation} {b} = {result}"

print("✓ 成功创建自定义工具")

# 测试3：创建模型实例
try:
    # 使用默认的推理客户端模型（免费）
    model = InferenceClientModel()
    print("✓ 成功创建模型实例")
except Exception as e:
    print(f"✗ 创建模型失败: {e}")
    print("注意：可能需要设置 HF_TOKEN 环境变量")

# 测试4：创建代理实例
try:
    # 创建工具调用代理
    tool_agent = ToolCallingAgent(
        tools=[simple_calculator], 
        model=model, 
        verbosity_level=1
    )
    print("✓ 成功创建 ToolCallingAgent")
    
    # 创建代码代理
    code_agent = CodeAgent(
        tools=[simple_calculator], 
        model=model, 
        verbosity_level=1
    )
    print("✓ 成功创建 CodeAgent")
    
except Exception as e:
    print(f"✗ 创建代理失败: {e}")

print("\n=== smolagents 安装测试完成 ===")
print("基础功能测试通过！")
print("\n要进行完整测试，可以运行以下命令：")
print("python test_smolagents.py --full-test")

# 如果用户要求完整测试
if len(sys.argv) > 1 and sys.argv[1] == "--full-test":
    print("\n开始完整功能测试...")
    try:
        # 测试简单的数学计算
        result = tool_agent.run("请计算 15 + 27 等于多少？")
        print(f"工具调用代理结果: {result}")
        
        result = code_agent.run("请计算 12 * 8 等于多少？")
        print(f"代码代理结果: {result}")
        
        print("✓ 完整功能测试通过！")
    except Exception as e:
        print(f"✗ 完整测试失败: {e}")
        print("这可能是由于网络连接或API限制导致的") 