#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络搜索功能调试脚本

用于调试WebSearchTool的搜索问题，特别是Bing RSS搜索返回0结果的问题
"""

import requests
import xml.etree.ElementTree as ET
import json
from urllib.parse import quote, urlencode


def debug_bing_rss_search(query="transformer是什么？"):
    """
    详细调试Bing RSS搜索功能
    """
    print("🔍 [调试] Bing RSS搜索详细分析")
    print("="*80)
    
    url = "https://www.bing.com/search"
    params = {"q": query, "format": "rss"}
    
    print(f"📋 请求信息:")
    print(f"  URL: {url}")
    print(f"  参数: {params}")
    print(f"  完整URL: {url}?{urlencode(params)}")
    
    try:
        # 添加更多的请求头，模拟真实浏览器
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=20)
        
        print(f"\n📤 响应信息:")
        print(f"  状态码: {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print(f"  Content-Length: {response.headers.get('Content-Length', 'N/A')}")
        print(f"  Server: {response.headers.get('Server', 'N/A')}")
        print(f"  响应长度: {len(response.text)} 字符")
        
        # 保存完整响应到文件以便分析
        with open("bing_response_debug.xml", "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"  💾 完整响应已保存到: bing_response_debug.xml")
        
        print(f"\n📄 响应内容 (完整):")
        print("-" * 80)
        print(response.text)
        print("-" * 80)
        
        # 尝试解析XML
        print(f"\n🔧 XML解析分析:")
        try:
            root = ET.fromstring(response.text)
            print(f"  ✅ XML解析成功")
            print(f"  根元素: {root.tag}")
            print(f"  根元素属性: {root.attrib}")
            
            # 打印完整的XML结构
            print(f"\n🌳 XML结构树:")
            def print_element(element, indent=0):
                spaces = "  " * indent
                print(f"{spaces}{element.tag}: {element.text[:100] if element.text else ''}")
                for child in element:
                    print_element(child, indent + 1)
            
            print_element(root)
            
            # 查找所有可能的元素
            print(f"\n🔍 搜索各种可能的内容元素:")
            
            # 尝试不同的XPath表达式
            xpath_patterns = [
                ".//item",
                ".//entry", 
                ".//result",
                ".//channel/item",
                ".//rss/channel/item",
                "./channel/item",
                ".//feed/entry"
            ]
            
            for pattern in xpath_patterns:
                items = root.findall(pattern)
                print(f"  {pattern}: 找到 {len(items)} 个元素")
                if len(items) > 0:
                    for i, item in enumerate(items[:3]):  # 只显示前3个
                        print(f"    项目 {i+1}: {item.tag} - {item.text[:50] if item.text else ''}")
            
            # 查找所有文本内容
            print(f"\n📝 所有包含文本的元素:")
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    print(f"  {elem.tag}: {elem.text.strip()[:100]}")
                    
        except ET.ParseError as e:
            print(f"  ❌ XML解析失败: {e}")
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None


def test_alternative_search_methods():
    """
    测试其他搜索方法
    """
    print("\n" + "="*80)
    print("🔧 测试替代搜索方法")
    print("="*80)
    
    query = "Python机器学习库"
    
    # 方法1: 直接HTML搜索
    print("\n1️⃣ 测试Bing HTML搜索:")
    try:
        url = "https://www.bing.com/search"
        params = {"q": query}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=20)
        print(f"  状态码: {response.status_code}")
        print(f"  响应长度: {len(response.text)} 字符")
        
        # 检查是否包含搜索结果
        if "li class" in response.text or "div class" in response.text:
            print(f"  ✅ 包含HTML结构，可能有搜索结果")
            # 保存HTML响应
            with open("bing_html_debug.html", "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"  💾 HTML响应已保存到: bing_html_debug.html")
        else:
            print(f"  ❌ 响应异常，可能被阻止")
            
    except Exception as e:
        print(f"  ❌ HTML搜索失败: {e}")
    
    # 方法2: DuckDuckGo搜索
    print("\n2️⃣ 测试DuckDuckGo搜索:")
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        response = requests.get(url, params=params, timeout=20)
        print(f"  状态码: {response.status_code}")
        
        data = response.json()
        print(f"  抽象结果: {len(data.get('Abstract', ''))} 字符")
        print(f"  相关主题: {len(data.get('RelatedTopics', []))} 个")
        print(f"  即时答案: {data.get('Answer', 'N/A')}")
        
        if data.get('RelatedTopics'):
            print(f"  ✅ 找到相关主题:")
            for i, topic in enumerate(data['RelatedTopics'][:3]):
                print(f"    {i+1}. {topic.get('Text', '')[:100]}")
                
    except Exception as e:
        print(f"  ❌ DuckDuckGo搜索失败: {e}")


def test_fixed_web_search_tool():
    """
    测试修复后的WebSearchTool
    """
    print("\n" + "="*80)
    print("🛠️ 测试修复的WebSearchTool实现")
    print("="*80)
    
    class FixedWebSearchTool:
        def __init__(self, engine="bing", max_results=10):
            self.engine = engine
            self.max_results = max_results
        
        def search_bing_html(self, query: str) -> list:
            """
            使用HTML解析的Bing搜索（更可靠）
            """
            print(f"🌐 使用HTML解析搜索: {query}")
            
            try:
                import re
                from bs4 import BeautifulSoup
                
                url = "https://www.bing.com/search"
                params = {"q": query}
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=20)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = []
                
                # 查找搜索结果元素
                result_elements = soup.find_all(['li', 'div'], class_=re.compile(r'b_algo|b_result'))
                
                print(f"  找到候选结果元素: {len(result_elements)} 个")
                
                for element in result_elements[:self.max_results]:
                    title_elem = element.find(['h2', 'h3', 'a'])
                    desc_elem = element.find(['p', 'div'], class_=re.compile(r'b_caption|caption'))
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        link = title_elem.get('href', '') if title_elem.name == 'a' else ''
                        if not link and title_elem.find('a'):
                            link = title_elem.find('a').get('href', '')
                        
                        description = desc_elem.get_text(strip=True) if desc_elem else ''
                        
                        if title and (link or description):
                            results.append({
                                'title': title,
                                'link': link,
                                'description': description
                            })
                
                print(f"  ✅ 成功提取 {len(results)} 个结果")
                return results
                
            except ImportError:
                print("  ❌ 需要安装BeautifulSoup: pip install beautifulsoup4")
                return []
            except Exception as e:
                print(f"  ❌ HTML搜索失败: {e}")
                return []
        
        def search_duckduckgo(self, query: str) -> list:
            """
            使用DuckDuckGo API搜索
            """
            print(f"🦆 使用DuckDuckGo搜索: {query}")
            
            try:
                url = "https://api.duckduckgo.com/"
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1"
                }
                
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                # 从相关主题中提取结果
                for topic in data.get('RelatedTopics', [])[:self.max_results]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'title': topic.get('Text', '')[:100],
                            'link': topic.get('FirstURL', ''),
                            'description': topic.get('Text', '')
                        })
                
                # 如果有抽象信息也添加进去
                if data.get('Abstract'):
                    results.insert(0, {
                        'title': f"关于 {query}",
                        'link': data.get('AbstractURL', ''),
                        'description': data.get('Abstract', '')
                    })
                
                print(f"  ✅ 成功提取 {len(results)} 个结果")
                return results
                
            except Exception as e:
                print(f"  ❌ DuckDuckGo搜索失败: {e}")
                return []
        
        def forward(self, query: str) -> str:
            """
            执行搜索并格式化结果
            """
            print(f"\n🔍 开始搜索: {query}")
            
            # 首先尝试DuckDuckGo（更稳定）
            results = self.search_duckduckgo(query)
            
            # 如果DuckDuckGo没有结果，尝试Bing HTML
            if not results:
                print("  🔄 尝试Bing HTML搜索...")
                results = self.search_bing_html(query)
            
            if not results:
                return f"❌ 搜索 '{query}' 没有找到结果"
            
            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_result = f"{i}. **{result['title']}**"
                if result['link']:
                    formatted_result += f"\n   链接: {result['link']}"
                if result['description']:
                    formatted_result += f"\n   描述: {result['description'][:200]}..."
                formatted_results.append(formatted_result)
            
            final_result = f"🔍 搜索 '{query}' 的结果:\n\n" + "\n\n".join(formatted_results)
            
            print(f"\n✅ 格式化结果长度: {len(final_result)} 字符")
            print(f"📄 结果预览:\n{final_result[:500]}...")
            
            return final_result
    
    # 测试修复的搜索工具
    tool = FixedWebSearchTool()
    result = tool.forward("Python机器学习库")
    
    print(f"\n🎯 最终搜索结果:")
    print("="*80)
    print(result)
    print("="*80)


def main():
    """
    主调试函数
    """
    print("🔧 网络搜索功能调试工具")
    print("="*80)
    
    # 1. 调试Bing RSS搜索
    debug_bing_rss_search()
    
    # 2. 测试替代搜索方法
    # test_alternative_search_methods()
    
    # 3. 测试修复的搜索工具
    # test_fixed_web_search_tool()
    
    print(f"\n✅ 调试完成！")
    print(f"📝 文件输出:")
    print(f"  - bing_response_debug.xml: Bing RSS响应")
    print(f"  - bing_html_debug.html: Bing HTML响应")


if __name__ == "__main__":
    main() 