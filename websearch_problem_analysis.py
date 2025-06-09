#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebSearchTool问题分析和解决方案

详细分析原始WebSearchTool中search_bing方法的问题并提供修复方案
"""

import requests
import xml.etree.ElementTree as ET


def analyze_original_problem():
    """
    分析原始WebSearchTool的问题
    """
    print("🔍 WebSearchTool问题分析")
    print("="*80)
    
    query = "Python机器学习库"
    
    # 1. 重现原始代码的问题
    print("\n1️⃣ 原始代码实现 (有问题的版本):")
    print("-" * 60)
    
    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "format": "rss"},
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    items = root.findall(".//item")
    
    print(f"找到items数量: {len(items)}")
    
    # 原始代码的问题实现
    results_original = [
        {
            "title": item.findtext("title"),
            "link": item.findtext("link"), 
            "description": item.findtext("description"),
        }
        for item in items[:3]  # 只检查前3个
    ]
    
    print("原始方法的结果:")
    for i, result in enumerate(results_original):
        print(f"  {i+1}. title: {result['title']}")
        print(f"     link: {result['link']}")
        print(f"     desc: {result['description'][:50] if result['description'] else 'None'}...")
    
    # 2. 分析为什么失败
    print(f"\n2️⃣ 问题分析:")
    print("-" * 60)
    
    if items:
        first_item = items[0]
        print(f"第一个item的标签: {first_item.tag}")
        print(f"第一个item的属性: {first_item.attrib}")
        print(f"第一个item的文本: {first_item.text}")
        
        print(f"\n第一个item的所有子元素:")
        for child in first_item:
            print(f"  - {child.tag}: {child.text}")
        
        # 检查title元素
        title_elem = first_item.find("title")
        print(f"\nfind('title')结果: {title_elem}")
        if title_elem is not None:
            print(f"title元素的文本: {title_elem.text}")
        
        # 检查所有可能的title查找方式
        title_findtext = first_item.findtext("title")
        title_direct = first_item.find("title")
        
        print(f"\nfindtext('title'): {title_findtext}")
        print(f"find('title'): {title_direct}")
        if title_direct is not None:
            print(f"find('title').text: {title_direct.text}")


def demonstrate_correct_solution():
    """
    展示正确的解决方案
    """
    print(f"\n3️⃣ 正确的解决方案:")
    print("-" * 60)
    
    query = "Python机器学习库"
    
    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "format": "rss"},
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    items = root.findall(".//item")
    
    # 修复的实现
    results_fixed = []
    for item in items[:3]:  # 只检查前3个
        title_elem = item.find("title")
        link_elem = item.find("link") 
        desc_elem = item.find("description")
        
        title = title_elem.text if title_elem is not None else None
        link = link_elem.text if link_elem is not None else None
        description = desc_elem.text if desc_elem is not None else None
        
        # 只有当title和link都存在时才添加结果
        if title and link:
            results_fixed.append({
                "title": title,
                "link": link,
                "description": description or "",
            })
    
    print("修复后的结果:")
    for i, result in enumerate(results_fixed):
        print(f"  {i+1}. title: {result['title']}")
        print(f"     link: {result['link']}")
        print(f"     desc: {result['description'][:50]}...")


def demonstrate_alternative_solution():
    """
    展示替代解决方案 - 直接文本搜索
    """
    print(f"\n4️⃣ 替代解决方案 - 直接文本提取:")
    print("-" * 60)
    
    query = "Python机器学习库"
    
    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "format": "rss"},
    )
    response.raise_for_status()
    
    # 使用正则表达式直接提取
    import re
    
    # 提取所有title标签的内容
    titles = re.findall(r'<title>(.*?)</title>', response.text)
    links = re.findall(r'<link>(.*?)</link>', response.text)  
    descriptions = re.findall(r'<description>(.*?)</description>', response.text)
    
    print(f"提取到的内容数量:")
    print(f"  标题: {len(titles)}")
    print(f"  链接: {len(links)}") 
    print(f"  描述: {len(descriptions)}")
    
    # 跳过第一个（通常是频道信息）
    if len(titles) > 1 and len(links) > 1:
        results_regex = []
        for i in range(1, min(4, len(titles))):  # 取前3个结果，跳过第一个
            if i < len(links) and i < len(descriptions):
                results_regex.append({
                    "title": titles[i],
                    "link": links[i], 
                    "description": descriptions[i] if i < len(descriptions) else ""
                })
        
        print(f"\n正则表达式提取的结果:")
        for i, result in enumerate(results_regex):
            print(f"  {i+1}. title: {result['title']}")
            print(f"     link: {result['link']}")
            print(f"     desc: {result['description'][:50]}...")


def provide_final_fix():
    """
    提供最终的修复代码
    """
    print(f"\n5️⃣ 推荐的最终修复方案:")
    print("-" * 60)
    
    fix_code = '''
def search_bing(self, query: str) -> list:
    import xml.etree.ElementTree as ET
    import requests

    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "format": "rss"},
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    items = root.findall(".//item")
    
    results = []
    for item in items[: self.max_results]:
        # 使用find()而不是findtext()来获得更好的控制
        title_elem = item.find("title")
        link_elem = item.find("link")
        desc_elem = item.find("description")
        
        title = title_elem.text if title_elem is not None else None
        link = link_elem.text if link_elem is not None else None  
        description = desc_elem.text if desc_elem is not None else ""
        
        # 确保title和link都存在才添加结果
        if title and link:
            results.append({
                "title": title,
                "link": link,
                "description": description,
            })
    
    return results
'''
    
    print("修复的核心要点:")
    print("1. 使用 find() 方法而不是 findtext() 方法")
    print("2. 检查元素是否为 None 再获取 text 属性")  
    print("3. 确保 title 和 link 都存在才添加到结果中")
    print("4. 为 description 提供默认空字符串")
    
    print(f"\n修复代码:")
    print(fix_code)


def main():
    """
    主函数
    """
    analyze_original_problem()
    demonstrate_correct_solution()
    demonstrate_alternative_solution()
    provide_final_fix()
    
    print(f"\n" + "="*80)
    print("🎯 总结：WebSearchTool的主要问题")
    print("="*80)
    print("❌ 问题: item.findtext() 方法在某些XML结构下返回None")
    print("✅ 解决: 使用 item.find().text 并检查元素存在性")
    print("💡 原因: XML解析时元素结构可能不是预期的直接子元素关系")
    print("📝 建议: 增加错误处理和元素存在性检查")


if __name__ == "__main__":
    main() 