#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的DeepSeek API测试脚本
"""

import os
from smolagents import OpenAIServerModel, ChatMessage

def test_deepseek():
    print("=== DeepSeek API 测试 ===")
    
    # 检查API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        print(f"✅ API密钥已设置: {api_key[:10]}...")
    else:
        print("❌ 未找到API密钥")
        return False
    
    try:
        # 创建模型
        print("\n正在创建DeepSeek模型...")
        model = OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=api_key,
        )
        
        # 测试简单对话
        print("正在测试API连接...")
        
        messages = [ChatMessage(role="user", content="你好！请简短回答你是什么模型？")]
        response = model.generate(messages)
        
        print("✅ API测试成功！")
        print(f"DeepSeek回应: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_deepseek()
    if success:
        print("\n🎉 DeepSeek API设置完成！您现在可以在smolagents中使用DeepSeek了。")
    else:
        print("\n❌ 请检查API密钥设置和网络连接。") 