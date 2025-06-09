#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
百度搜索工具使用示例

本示例展示如何使用 smolagents 中的百度搜索功能，包括：
1. 使用专门的 BaiduSearchTool
2. 使用通用的 WebSearchTool 配置为百度搜索
3. 中文搜索查询示例

作者: smolagents 团队
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.smolagents.default_tools import BaiduSearchTool, WebSearchTool


def demo_baidu_search_tool():
    """演示 BaiduSearchTool 的使用"""
    print("=" * 60)
    print("🔍 百度搜索工具 (BaiduSearchTool) 示例")
    print("=" * 60)
    
    # 创建百度搜索工具实例
    baidu_tool = BaiduSearchTool(max_results=5)
    
    # 测试搜索查询
    test_queries = [
        "人工智能发展趋势",
        "Python 编程教程",
        "机器学习入门",
        "北京天气"
    ]
    
    for query in test_queries:
        print(f"\n🔎 搜索查询: {query}")
        print("-" * 40)
        
        try:
            results = baidu_tool.forward(query)
            print(results)
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
        
        print("\n" + "="*60)


def demo_web_search_tool_baidu():
    """演示使用 WebSearchTool 进行百度搜索"""
    print("=" * 60)
    print("🌐 通用搜索工具 (WebSearchTool) - 百度引擎示例")
    print("=" * 60)
    
    # 创建配置为百度搜索的通用搜索工具
    web_tool = WebSearchTool(max_results=3, engine="baidu")
    
    # 测试搜索
    query = "深度学习框架比较"
    print(f"\n🔎 搜索查询: {query}")
    print("-" * 40)
    
    try:
        results = web_tool.forward(query)
        print(results)
    except Exception as e:
        print(f"❌ 搜索失败: {e}")


def compare_search_engines():
    """比较不同搜索引擎的结果"""
    print("=" * 60)
    print("⚖️  搜索引擎比较示例")
    print("=" * 60)
    
    query = "开源大语言模型"
    engines = ["duckduckgo", "bing", "baidu"]
    
    for engine in engines:
        print(f"\n🔍 使用 {engine.upper()} 搜索: {query}")
        print("-" * 40)
        
        try:
            tool = WebSearchTool(max_results=2, engine=engine)
            results = tool.forward(query)
            print(results[:300] + "..." if len(results) > 300 else results)
        except Exception as e:
            print(f"❌ {engine} 搜索失败: {e}")
        
        print()


def main():
    """主函数"""
    print("🚀 百度搜索工具示例程序启动")
    print("本示例将演示如何使用百度搜索功能")
    print()
    
    try:
        # 示例1: 专用百度搜索工具
        demo_baidu_search_tool()
        
        # 示例2: 通用搜索工具配置为百度
        demo_web_search_tool_baidu()
        
        # 示例3: 比较不同搜索引擎
        compare_search_engines()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序运行出错: {e}")
    finally:
        print("\n✅ 示例程序结束")


if __name__ == "__main__":
    main() 