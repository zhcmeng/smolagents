# smolagents中的DeepSeek支持指南

本指南详细说明了如何在smolagents项目中集成和使用DeepSeek模型。

## 概述

smolagents项目采用灵活的模型架构，已经内置支持多种方式来使用DeepSeek模型：

1. **OpenAIServerModel**（推荐）- 利用DeepSeek的OpenAI兼容API
2. **LiteLLMModel** - 通过LiteLLM库访问DeepSeek
3. **InferenceClientModel** - 通过HuggingFace推理提供商
4. **自定义DeepSeekModel类** - 专门的DeepSeek模型实现

## DeepSeek API信息

### 基础信息
- **API基础URL**: `https://api.deepseek.com` 或 `https://api.deepseek.com/v1`
- **兼容性**: 完全兼容OpenAI API格式
- **主要模型**:
  - `deepseek-chat`: DeepSeek-V3-0324（通用对话模型）
  - `deepseek-reasoner`: DeepSeek-R1-0528（推理模型）

### 获取API密钥
1. 访问 [DeepSeek平台](https://platform.deepseek.com/api_keys)
2. 注册账户并申请API密钥
3. 设置环境变量：`export DEEPSEEK_API_KEY=your_api_key`

## 集成方法

### 方法1: OpenAIServerModel（推荐）

这是最直接的方式，因为DeepSeek提供了完全兼容OpenAI的API：

```python
from smolagents import OpenAIServerModel, CodeAgent
from smolagents.default_tools import PythonInterpreterTool, WebSearchTool

# DeepSeek Chat模型
model = OpenAIServerModel(
    model_id="deepseek-chat",
    api_base="https://api.deepseek.com",
    api_key="your_deepseek_api_key",
)

# 创建智能体
agent = CodeAgent(
    tools=[PythonInterpreterTool(), WebSearchTool()],
    model=model,
    verbose=1
)

# 使用智能体
result = agent.run("请分析这个数据集并生成可视化图表")
```

#### DeepSeek推理模型

```python
# DeepSeek推理模型 - 适合复杂推理任务
model = OpenAIServerModel(
    model_id="deepseek-reasoner",
    api_base="https://api.deepseek.com",
    api_key="your_deepseek_api_key",
    reasoning_effort="high",  # 推理强度: "low", "medium", "high", "none"
)

agent = CodeAgent(
    tools=[PythonInterpreterTool()],
    model=model,
    verbose=2
)

result = agent.run("解决这个复杂的数学优化问题...")
```

### 方法2: LiteLLMModel

通过LiteLLM库访问DeepSeek：

```python
from smolagents import LiteLLMModel, CodeAgent

model = LiteLLMModel(
    model_id="deepseek/deepseek-chat",  # LiteLLM格式
    api_key="your_deepseek_api_key",
    temperature=0.7,
    max_tokens=4096,
)

agent = CodeAgent(
    tools=[PythonInterpreterTool()],
    model=model
)

result = agent.run("编写一个机器学习模型训练脚本")
```

### 方法3: InferenceClientModel

如果DeepSeek在HuggingFace推理提供商中可用：

```python
from smolagents import InferenceClientModel, CodeAgent

# 通过第三方提供商访问DeepSeek
model = InferenceClientModel(
    model_id="deepseek-ai/DeepSeek-R1",
    provider="together",  # 或其他支持DeepSeek的提供商
    token="your_hf_or_provider_token",
)

agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model
)
```

### 方法4: 自定义DeepSeekModel类

创建专门的DeepSeek模型类：

```python
from smolagents import OpenAIServerModel, CodeAgent
import os

class DeepSeekModel(OpenAIServerModel):
    """专门为DeepSeek优化的模型类"""
    
    def __init__(self, model_type="chat", api_key=None, **kwargs):
        model_mapping = {
            "chat": "deepseek-chat",
            "reasoner": "deepseek-reasoner"
        }
        
        super().__init__(
            model_id=model_mapping.get(model_type, "deepseek-chat"),
            api_base="https://api.deepseek.com",
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            **kwargs
        )
        
        self.model_type = model_type

# 使用自定义类
model = DeepSeekModel(
    model_type="chat",
    temperature=0.7
)

agent = CodeAgent(
    tools=[PythonInterpreterTool()],
    model=model
)
```

## CLI使用方法

### 基础CLI使用

使用原有的CLI：

```bash
# 使用OpenAIServerModel方式
python -m smolagents.cli "解释量子计算原理" \
  --model-type OpenAIServerModel \
  --model-id deepseek-chat \
  --api-base https://api.deepseek.com \
  --api-key $DEEPSEEK_API_KEY \
  --tools web_search python_interpreter

# 使用LiteLLM方式
python -m smolagents.cli "编写排序算法" \
  --model-type LiteLLMModel \
  --model-id deepseek/deepseek-chat \
  --api-key $DEEPSEEK_API_KEY \
  --tools python_interpreter
```

### 扩展CLI使用

如果使用我们的扩展CLI（`cli_extended.py`）：

```bash
# 使用专门的DeepSeek模型类型
python cli_extended.py "编写一个Web爬虫" \
  --model-type DeepSeekModel \
  --deepseek-model-type chat

# 使用推理模型进行复杂推理
python cli_extended.py "解决数学难题" \
  --model-type DeepSeekModel \
  --deepseek-model-type reasoner \
  --reasoning-effort high

# 通过OpenAI兼容接口
python cli_extended.py "数据分析任务" \
  --model-type OpenAIServerModel \
  --model-id deepseek-chat \
  --api-key $DEEPSEEK_API_KEY
```

## 最佳实践

### 1. 环境配置

```bash
# 设置DeepSeek API密钥
export DEEPSEEK_API_KEY=your_deepseek_api_key

# 可选：设置其他相关环境变量
export HF_TOKEN=your_huggingface_token
```

### 2. 模型选择指南

- **deepseek-chat**: 适合通用对话、代码生成、文本处理等任务
- **deepseek-reasoner**: 适合复杂推理、数学问题、逻辑分析等任务

### 3. 推理强度设置

对于推理模型，可以调整推理强度：
- `none`: 禁用推理（如果模型支持）
- `low`: 低推理强度，响应更快
- `medium`: 中等推理强度（默认）
- `high`: 高推理强度，更深入的推理

### 4. 错误处理

```python
import os
from smolagents import OpenAIServerModel, CodeAgent

try:
    # 检查API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY环境变量未设置")
    
    model = OpenAIServerModel(
        model_id="deepseek-chat",
        api_base="https://api.deepseek.com",
        api_key=api_key,
    )
    
    agent = CodeAgent(tools=[], model=model)
    result = agent.run("你的任务描述")
    
except Exception as e:
    print(f"错误: {e}")
    print("请检查：")
    print("1. DEEPSEEK_API_KEY是否正确设置")
    print("2. 网络连接是否正常")
    print("3. API密钥是否有效")
```

## 性能优化建议

### 1. 流式响应

对于长文本生成，启用流式响应：

```python
model = OpenAIServerModel(
    model_id="deepseek-chat",
    api_base="https://api.deepseek.com",
    api_key="your_api_key",
    stream=True,  # 启用流式响应
)
```

### 2. 参数调优

```python
model = OpenAIServerModel(
    model_id="deepseek-chat",
    api_base="https://api.deepseek.com",
    api_key="your_api_key",
    temperature=0.7,     # 控制随机性
    max_tokens=4096,     # 最大输出长度
    top_p=0.9,          # 核采样参数
)
```

### 3. 缓存和重试

```python
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="deepseek-chat",
    api_base="https://api.deepseek.com",
    api_key="your_api_key",
    client_kwargs={
        "max_retries": 3,      # 重试次数
        "timeout": 60,         # 超时时间
    }
)
```

## 常见问题解决

### Q: 如何处理API密钥错误？
A: 确保设置了正确的环境变量：
```bash
export DEEPSEEK_API_KEY=your_actual_api_key
```

### Q: 如何选择合适的模型？
A: 
- 通用任务使用 `deepseek-chat`
- 需要深度推理的任务使用 `deepseek-reasoner`

### Q: 如何提高响应速度？
A: 
- 降低 `max_tokens` 设置
- 对推理模型使用较低的 `reasoning_effort`
- 启用流式响应

### Q: 支持哪些工具？
A: DeepSeek模型支持smolagents的所有标准工具：
- PythonInterpreterTool
- WebSearchTool
- 自定义工具

## 扩展开发

### 添加新的CLI支持

要在原有CLI中添加DeepSeek支持，可以修改 `load_model` 函数：

```python
def load_model(model_type, model_id, api_base=None, api_key=None, provider=None):
    if model_type == "OpenAIServerModel":
        # 检查是否是DeepSeek模型
        if model_id in ["deepseek-chat", "deepseek-reasoner"]:
            return OpenAIServerModel(
                model_id=model_id,
                api_base=api_base or "https://api.deepseek.com",
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            )
        # 原有逻辑...
```

### 创建专门的DeepSeek工具

```python
from smolagents import Tool

class DeepSeekAnalysisTool(Tool):
    name = "deepseek_analysis"
    description = "使用DeepSeek模型进行深度分析"
    
    def forward(self, query: str):
        # 实现特定的DeepSeek分析逻辑
        pass
```

## 总结

smolagents项目已经具备了完整的DeepSeek支持能力，主要通过以下方式实现：

1. **现有架构支持**: 通过OpenAIServerModel可以直接使用DeepSeek的OpenAI兼容API
2. **多种集成方式**: 支持LiteLLM、InferenceClient等多种访问方式
3. **灵活配置**: 支持不同的模型类型和推理强度设置
4. **CLI友好**: 可以通过命令行直接使用
5. **扩展性强**: 可以轻松创建自定义DeepSeek模型类

这种设计使得用户可以根据自己的需求选择最适合的集成方式，无需对smolagents的核心架构进行重大修改。 