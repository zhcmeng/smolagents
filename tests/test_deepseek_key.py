#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试DeepSeek API密钥设置的脚本
"""

import os
from smolagents import OpenAIServerModel

def test_deepseek_api_key():
    """测试DeepSeek API密钥是否正确设置"""
    
    print("=== DeepSeek API密钥测试 ===\n")
    
    # 检查环境变量
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("❌ 错误: 未找到DEEPSEEK_API_KEY环境变量")
        print("\n请设置API密钥：")
        print("Linux/macOS: export DEEPSEEK_API_KEY=your_api_key")
        print("Windows: set DEEPSEEK_API_KEY=your_api_key")
        return False
    
    print(f"✅ 找到API密钥: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else ''}")
    
    # 测试API连接
    try:
        print("\n正在测试API连接...")
        
        model = OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=api_key,
        )
        
        # 发送一个简单的测试请求
        from smolagents.types import ChatMessage
        test_message = ChatMessage(role="user", content="Hello!")
        
        response = model.generate([test_message])
        
        print("✅ API连接成功!")
        print(f"测试响应: {response.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        print("\n可能的问题:")
        print("1. API密钥无效或已过期")
        print("2. 网络连接问题")
        print("3. API服务暂时不可用")
        return False

def test_different_models():
    """测试不同的DeepSeek模型"""
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("请先设置DEEPSEEK_API_KEY环境变量")
        return
    
    models_to_test = [
        ("deepseek-chat", "DeepSeek Chat模型"),
        ("deepseek-reasoner", "DeepSeek推理模型")
    ]
    
    print("\n=== 测试不同模型 ===")
    
    for model_id, description in models_to_test:
        try:
            print(f"\n测试 {description} ({model_id})...")
            
            model = OpenAIServerModel(
                model_id=model_id,
                api_base="https://api.deepseek.com",
                api_key=api_key,
            )
            
            from smolagents.types import ChatMessage
            test_message = ChatMessage(role="user", content="简短回答：你是什么模型？")
            
            response = model.generate([test_message])
            print(f"✅ {description}响应: {response.content[:50]}...")
            
        except Exception as e:
            print(f"❌ {description}测试失败: {e}")

def show_setup_instructions():
    """显示详细的设置说明"""
    
    print("\n=== DeepSeek API密钥设置指南 ===")
    print("\n1. 获取API密钥:")
    print("   - 访问: https://platform.deepseek.com/api_keys")
    print("   - 注册账户并申请API密钥")
    
    print("\n2. 设置环境变量:")
    print("   Linux/macOS:")
    print("   export DEEPSEEK_API_KEY=your_api_key")
    print("   echo 'export DEEPSEEK_API_KEY=your_api_key' >> ~/.bashrc")
    
    print("\n   Windows PowerShell:")
    print("   $env:DEEPSEEK_API_KEY='your_api_key'")
    
    print("\n   Windows CMD:")
    print("   set DEEPSEEK_API_KEY=your_api_key")
    
    print("\n3. 或创建.env文件:")
    print("   DEEPSEEK_API_KEY=your_api_key")
    
    print("\n4. 在smolagents中使用:")
    print("""
from smolagents import OpenAIServerModel, CodeAgent

model = OpenAIServerModel(
    model_id="deepseek-chat",
    api_base="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

agent = CodeAgent(tools=[], model=model)
result = agent.run("你的问题")
""")

if __name__ == "__main__":
    # 运行测试
    success = test_deepseek_api_key()
    
    if success:
        # 如果基本测试成功，测试不同模型
        test_different_models()
        
        print("\n🎉 DeepSeek API密钥设置成功！")
        print("你现在可以在smolagents中使用DeepSeek模型了。")
    else:
        # 如果测试失败，显示设置说明
        show_setup_instructions()