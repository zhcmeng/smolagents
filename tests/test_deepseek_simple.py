#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的DeepSeek测试，用于调试问题
"""

import os

def test_basic_imports():
    """测试基础导入"""
    try:
        from smolagents import OpenAIServerModel, ChatMessage
        print("✅ 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_chatmessage_creation():
    """测试ChatMessage创建"""
    try:
        from smolagents import ChatMessage
        message = ChatMessage(role="user", content="Hello!")
        print(f"✅ ChatMessage创建成功: {message}")
        print(f"   role: {message.role}")
        print(f"   content: {message.content}")
        return True
    except Exception as e:
        print(f"❌ ChatMessage创建失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    try:
        from smolagents import OpenAIServerModel
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 未找到DEEPSEEK_API_KEY环境变量")
            return False
        
        model = OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=api_key,
        )
        print("✅ 模型创建成功")
        return True
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_simple_generation():
    """测试简单生成"""
    try:
        from smolagents import OpenAIServerModel
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 未找到DEEPSEEK_API_KEY环境变量")
            return False
        
        print("创建模型...")
        model = OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=api_key,
        )
        
        print("创建消息（字典格式）...")
        # 使用字典格式而不是ChatMessage对象
        messages = [{"role": "user", "content": "Hello!"}]
        print(f"消息内容: {messages}")
        
        print("发送请求...")
        response = model.generate(messages)
        
        print("✅ 生成成功!")
        print(f"响应: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== DeepSeek简单测试 ===")
    
    print("\n1. 测试导入...")
    if not test_basic_imports():
        exit(1)
    
    print("\n2. 测试ChatMessage创建...")
    if not test_chatmessage_creation():
        exit(1)
    
    print("\n3. 测试模型创建...")
    if not test_model_creation():
        exit(1)
    
    print("\n4. 测试简单生成...")
    if test_simple_generation():
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 生成测试失败") 