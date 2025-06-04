#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSeek模型的综合测试套件

复用现有测试逻辑，专门针对DeepSeek模型进行测试
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from smolagents import OpenAIServerModel, CodeAgent, LiteLLMModel
from smolagents.default_tools import PythonInterpreterTool, WebSearchTool, FinalAnswerTool
from smolagents.types import ChatMessage, MessageRole
from smolagents.cli import load_model

from .utils.markers import require_run_all


class TestDeepSeekModels:
    """测试DeepSeek模型的基础功能"""
    
    @pytest.fixture
    def deepseek_api_key(self):
        """获取DeepSeek API密钥"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY环境变量未设置")
        return api_key
    
    @pytest.fixture  
    def deepseek_chat_model(self, deepseek_api_key):
        """DeepSeek Chat模型fixture"""
        return OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=deepseek_api_key,
        )
    
    @pytest.fixture
    def deepseek_reasoner_model(self, deepseek_api_key):
        """DeepSeek推理模型fixture"""
        return OpenAIServerModel(
            model_id="deepseek-reasoner", 
            api_base="https://api.deepseek.com",
            api_key=deepseek_api_key,
            reasoning_effort="medium",
        )
    
    def test_deepseek_chat_basic_generation(self, deepseek_chat_model):
        """测试DeepSeek Chat模型的基础生成"""
        messages = [ChatMessage(role="user", content="简短回答：1+1等于几？")]
        response = deepseek_chat_model.generate(messages)
        
        assert response.content is not None
        assert len(response.content.strip()) > 0
        assert "2" in response.content
    
    def test_deepseek_reasoner_basic_generation(self, deepseek_reasoner_model):
        """测试DeepSeek推理模型的基础生成"""
        messages = [ChatMessage(role="user", content="简短回答：什么是人工智能？")]
        response = deepseek_reasoner_model.generate(messages)
        
        assert response.content is not None
        assert len(response.content.strip()) > 0
    
    @require_run_all
    def test_deepseek_streaming(self, deepseek_chat_model):
        """测试DeepSeek流式生成"""
        messages = [ChatMessage(role="user", content="请简短介绍Python")]
        
        content_parts = []
        for chunk in deepseek_chat_model.generate_stream(messages):
            if chunk.content:
                content_parts.append(chunk.content)
        
        full_content = "".join(content_parts)
        assert len(full_content) > 0
        assert "python" in full_content.lower() or "Python" in full_content
    
    def test_deepseek_with_stop_sequences(self, deepseek_chat_model):
        """测试DeepSeek停止序列功能"""
        messages = [ChatMessage(role="user", content="计算1+1，然后说'结束'")]
        response = deepseek_chat_model.generate(messages, stop_sequences=["结束"])
        
        assert response.content is not None
        assert "结束" not in response.content
    
    def test_deepseek_client_kwargs(self, deepseek_api_key):
        """测试DeepSeek客户端参数传递（复用OpenAI测试逻辑）"""
        with patch("openai.OpenAI") as MockOpenAI:
            model = OpenAIServerModel(
                model_id="deepseek-chat",
                api_base="https://api.deepseek.com", 
                api_key=deepseek_api_key,
                client_kwargs={"max_retries": 3, "timeout": 30}
            )
        
        MockOpenAI.assert_called_once_with(
            base_url="https://api.deepseek.com",
            api_key=deepseek_api_key,
            organization=None,
            project=None,
            max_retries=3,
            timeout=30
        )


class TestDeepSeekAgents:
    """测试DeepSeek模型与智能体的集成"""
    
    @pytest.fixture
    def deepseek_model(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY环境变量未设置")
        return OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=api_key,
        )
    
    def test_deepseek_code_agent_basic(self, deepseek_model):
        """测试DeepSeek与CodeAgent的基础集成"""
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=deepseek_model,
            verbose=0
        )
        
        # 测试简单的数学计算
        result = agent.run("计算3的阶乘")
        assert result is not None
    
    def test_deepseek_code_agent_with_web_search(self, deepseek_model):
        """测试DeepSeek与网络搜索工具的集成"""
        agent = CodeAgent(
            tools=[WebSearchTool()],
            model=deepseek_model,
            verbose=0
        )
        
        # 测试网络搜索功能
        result = agent.run("搜索Python最新版本信息")
        assert result is not None
    
    @require_run_all
    def test_deepseek_tool_calling(self, deepseek_model):
        """测试DeepSeek的工具调用功能（复用OpenAI测试逻辑）"""
        messages = [
            ChatMessage(
                role="user", 
                content="请返回最终答案'测试成功'"
            )
        ]
        
        # 测试工具调用是否正常工作
        for chunk in deepseek_model.generate_stream(messages, tools_to_call_from=[FinalAnswerTool()]):
            if chunk.tool_calls:
                assert chunk.tool_calls[0].function.name == "final_answer"
                break


class TestDeepSeekCLI:
    """测试DeepSeek与CLI的集成"""
    
    @pytest.fixture
    def set_deepseek_env(self, monkeypatch):
        """设置DeepSeek环境变量"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)
        else:
            monkeypatch.setenv("DEEPSEEK_API_KEY", "test_deepseek_key")
    
    def test_load_deepseek_model_openai_server(self, set_deepseek_env):
        """测试通过OpenAIServerModel加载DeepSeek"""
        with patch("openai.OpenAI") as MockOpenAI:
            model = load_model(
                "OpenAIServerModel", 
                "deepseek-chat",
                api_base="https://api.deepseek.com",
                api_key="test_deepseek_key"
            )
        
        assert isinstance(model, OpenAIServerModel)
        assert model.model_id == "deepseek-chat"
        MockOpenAI.assert_called_once()
        assert MockOpenAI.call_args.kwargs["base_url"] == "https://api.deepseek.com"
    
    def test_load_deepseek_model_litellm(self):
        """测试通过LiteLLM加载DeepSeek"""
        model = load_model(
            "LiteLLMModel",
            "deepseek/deepseek-chat", 
            api_key="test_api_key"
        )
        
        assert isinstance(model, LiteLLMModel)
        assert model.model_id == "deepseek/deepseek-chat"
        assert model.api_key == "test_api_key"


class TestDeepSeekLiteLLM:
    """测试DeepSeek与LiteLLM的集成"""
    
    def test_deepseek_litellm_model_creation(self):
        """测试创建DeepSeek LiteLLM模型"""
        model = LiteLLMModel(
            model_id="deepseek/deepseek-chat",
            api_key="test_key",
            temperature=0.7
        )
        
        assert model.model_id == "deepseek/deepseek-chat"
        assert model.api_key == "test_key"
    
    def test_deepseek_litellm_without_key(self):
        """测试没有API密钥时的LiteLLM DeepSeek模型"""
        model = LiteLLMModel(model_id="deepseek/deepseek-chat")
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        
        with pytest.raises(Exception) as e:
            model.generate(messages)
        # 应该因为缺少API密钥而失败
        assert "api_key" in str(e.value).lower() or "key" in str(e.value).lower()


class TestDeepSeekAdvanced:
    """DeepSeek高级功能测试"""
    
    @pytest.fixture
    def deepseek_model(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY环境变量未设置")
        return OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=api_key,
        )
    
    def test_deepseek_reasoning_effort_parameter(self):
        """测试DeepSeek推理强度参数"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY环境变量未设置")
        
        # 测试不同的推理强度设置
        for effort in ["low", "medium", "high"]:
            model = OpenAIServerModel(
                model_id="deepseek-reasoner",
                api_base="https://api.deepseek.com", 
                api_key=api_key,
                reasoning_effort=effort
            )
            assert model is not None
    
    def test_deepseek_custom_role_conversions(self, deepseek_model):
        """测试自定义角色转换"""
        # 测试角色转换功能
        custom_conversions = {MessageRole.USER: MessageRole.SYSTEM}
        model = OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            custom_role_conversions=custom_conversions
        )
        
        assert model.custom_role_conversions == custom_conversions


# 运行特定的DeepSeek测试
@pytest.mark.deepseek
class TestDeepSeekIntegration:
    """DeepSeek集成测试标记"""
    
    def test_deepseek_end_to_end(self):
        """端到端DeepSeek测试"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("需要DEEPSEEK_API_KEY进行端到端测试")
        
        # 创建模型
        model = OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=api_key,
        )
        
        # 创建智能体
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=model,
            verbose=1
        )
        
        # 执行任务
        result = agent.run("计算斐波那契数列的前5项")
        
        # 验证结果
        assert result is not None
        assert isinstance(result, str)
        assert len(result.strip()) > 0


if __name__ == "__main__":
    # 可以单独运行此文件进行测试
    import subprocess
    import sys
    
    # 运行DeepSeek特定的测试
    subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "-m", "deepseek"
    ]) 