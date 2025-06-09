"""
内存管理模块 - smolagents 的代理记忆系统

本模块实现了代理的内存管理功能，用于跟踪和存储代理执行过程中的所有步骤和状态。
代理的记忆系统是其智能行为的核心组件，记录了从任务开始到完成的完整执行轨迹。

主要功能:
- 记录代理的每个执行步骤和决策过程
- 管理工具调用历史和执行结果
- 存储观察结果、错误信息和时间统计
- 支持记忆回放和调试分析
- 提供消息格式转换和序列化功能
- 支持不同类型的步骤（行动、规划、任务等）

核心设计理念:
- 完整性：记录执行过程的每个细节
- 可追溯性：支持步骤回放和问题诊断
- 灵活性：支持多种步骤类型和扩展
- 高效性：优化内存使用和访问性能

主要类：
- AgentMemory: 代理记忆的主要管理类，统一管理所有步骤
- MemoryStep: 记忆步骤的抽象基类
- ActionStep: 行动步骤记录，包含工具调用和观察结果
- PlanningStep: 规划步骤记录，包含计划制定过程
- TaskStep: 任务步骤记录，标记新任务的开始
- SystemPromptStep: 系统提示步骤，定义代理行为模式
- FinalAnswerStep: 最终答案步骤，标记任务完成
- ToolCall: 工具调用记录，包含调用参数和结果
- Message: 消息类型定义，用于格式化通信

作者: HuggingFace 团队
版本: 1.0
"""

from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, TypedDict

from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


logger = getLogger(__name__)


class Message(TypedDict):
    """
    消息类型定义 - 标准化的消息格式
    
    定义了代理系统中使用的标准消息格式，确保不同组件之间
    的通信一致性和类型安全。
    
    属性:
        role (MessageRole): 消息发送者的角色（用户、助手、系统等）
        content (str | list[dict]): 消息内容，可以是简单文本或复杂的多媒体内容列表
    
    使用示例:
        message = Message(
            role=MessageRole.USER,
            content=[{"type": "text", "text": "你好"}]
        )
    """
    role: MessageRole
    content: str | list[dict[str, Any]]


@dataclass
class ToolCall:
    """
    工具调用记录类 - 记录工具调用的详细信息
    
    该类用于记录代理执行过程中的工具调用，包括调用的工具名称、
    传递的参数和调用标识符。这是代理行为分析的重要组成部分。
    
    属性:
        name (str): 被调用的工具名称
        arguments (Any): 传递给工具的参数，可以是任何类型
        id (str): 工具调用的唯一标识符
    
    使用示例:
        tool_call = ToolCall(
            name="calculator",
            arguments={"operation": "add", "a": 1, "b": 2},
            id="call_123"
        )
    """
    name: str
    arguments: Any
    id: str

    def dict(self):
        """
        转换为标准字典格式
        
        将工具调用对象转换为符合 OpenAI 工具调用格式的字典，
        便于与各种 LLM API 兼容和序列化存储。
        
        返回:
            dict: 包含工具调用信息的标准格式字典
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    """
    记忆步骤抽象基类 - 所有步骤类型的通用接口
    
    该类定义了代理记忆中所有步骤类型的通用接口和行为。
    所有具体的步骤类型都应该继承此类并实现相应的方法。
    
    主要作用:
    - 提供统一的步骤接口
    - 定义序列化和消息转换的标准
    - 确保所有步骤类型的一致性
    """
    
    def dict(self):
        """
        转换为字典格式
        
        将步骤对象转换为字典格式，便于序列化、存储和传输。
        默认实现使用 dataclasses.asdict 进行转换。
        
        返回:
            dict: 包含步骤所有字段的字典
        """
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        """
        转换为消息列表
        
        将步骤内容转换为消息格式，用于与 LLM 进行交互。
        子类必须实现此方法以定义具体的转换逻辑。
        
        参数:
            summary_mode (bool): 是否使用摘要模式，影响输出的详细程度
            
        返回:
            list[Message]: 转换后的消息列表
            
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    """
    行动步骤记录类 - 记录代理的具体行动和执行结果
    
    该类是代理记忆系统的核心组件，记录了代理执行的每一个具体行动，
    包括模型的输入输出、工具调用、观察结果、错误信息等完整信息。
    
    主要功能:
    - 记录完整的执行上下文和结果
    - 支持错误追踪和调试分析
    - 提供时间和令牌使用统计
    - 支持多媒体内容（如图像）的处理
    
    属性:
        step_number (int): 步骤编号，标识执行顺序
        timing (Timing): 执行时间统计信息
        model_input_messages (list[Message] | None): 输入给模型的消息列表
        tool_calls (list[ToolCall] | None): 本步骤中的工具调用列表
        error (AgentError | None): 执行过程中出现的错误
        model_output_message (ChatMessage | None): 模型的原始输出消息
        model_output (str | None): 模型输出的文本内容
        observations (str | None): 工具执行后的观察结果
        observations_images (list[PIL.Image.Image] | None): 观察结果中的图像
        action_output (Any): 行动的最终输出结果
        token_usage (TokenUsage | None): 令牌使用统计
    """
    step_number: int
    timing: Timing
    model_input_messages: list[Message] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None

    def dict(self):
        """
        转换为字典格式（覆盖基类方法）
        
        提供专门的字典转换逻辑，正确处理工具调用和行动输出的序列化。
        确保复杂对象能够正确转换为可序列化的格式。
        
        返回:
            dict: 包含所有步骤信息的字典，其中复杂对象已正确序列化
        """
        # 我们重写此方法以手动解析 tool_calls 和 action_output
        return {
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "timing": self.timing.dict(),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "step": self.step_number,
            "error": self.error.dict() if self.error else None,
            "model_output_message": self.model_output_message.dict() if self.model_output_message else None,
            "model_output": self.model_output,
            "observations": self.observations,
            "action_output": make_json_serializable(self.action_output),
        }

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        """
        转换为消息列表
        
        将行动步骤转换为一系列消息，用于重建与 LLM 的对话历史。
        根据步骤内容生成相应的消息，包括模型输出、工具调用、观察结果等。
        
        参数:
            summary_mode (bool): 是否使用摘要模式。在摘要模式下，某些详细信息会被省略
            
        返回:
            list[Message]: 表示此步骤的消息列表
        """
        messages = []
        
        # 添加模型输出消息（在非摘要模式下）
        if self.model_output is not None and not summary_mode:
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        # 添加工具调用消息
        if self.tool_calls is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        # 添加观察结果中的图像
        if self.observations_images:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        # 添加文本观察结果
        if self.observations is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
            
        # 添加错误信息
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                Message(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    """
    规划步骤记录类 - 记录代理的规划和思考过程
    
    该类用于记录代理进行任务规划的步骤，包括模型的输入输出、
    生成的计划内容以及相关的时间和令牌统计。
    
    规划步骤的特点:
    - 通常在执行开始时或定期进行
    - 帮助代理制定执行策略
    - 提供任务分解和步骤安排
    - 在摘要模式下可能被省略以避免重复
    
    属性:
        model_input_messages (list[Message]): 输入给模型的消息列表
        model_output_message (ChatMessage): 模型的输出消息
        plan (str): 生成的计划内容
        timing (Timing): 规划步骤的执行时间统计
        token_usage (TokenUsage | None): 令牌使用统计
    """
    model_input_messages: list[Message]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        """
        转换为消息列表
        
        将规划步骤转换为消息格式。在摘要模式下，规划消息会被省略
        以避免影响后续的规划过程。
        
        参数:
            summary_mode (bool): 是否使用摘要模式
            
        返回:
            list[Message]: 表示规划步骤的消息列表，摘要模式下返回空列表
        """
        if summary_mode:
            return []
        return [
            Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            Message(role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]),
            # 第二条消息创建角色变换，防止模型简单地继续规划消息
        ]


@dataclass
class TaskStep(MemoryStep):
    """
    任务步骤记录类 - 记录新任务的开始
    
    该类用于标记新任务的开始，记录任务的具体内容和相关的图像资源。
    通常是代理执行流程的第一个步骤。
    
    主要作用:
    - 明确定义任务目标和要求
    - 记录任务相关的多媒体资源
    - 为后续步骤提供上下文信息
    - 支持任务的重放和分析
    
    属性:
        task (str): 任务的具体描述内容
        task_images (list[PIL.Image.Image] | None): 任务相关的图像列表
    """
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        """
        转换为消息列表
        
        将任务步骤转换为用户消息，包含任务文本和相关图像。
        这个消息通常作为对话的开始。
        
        参数:
            summary_mode (bool): 摘要模式标志（对任务步骤无影响）
            
        返回:
            list[Message]: 包含任务信息的用户消息列表
        """
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        
        # 添加任务相关的图像
        if self.task_images:
            for image in self.task_images:
                content.append({"type": "image", "image": image})

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    """
    系统提示步骤记录类 - 记录代理的系统级配置
    
    该类用于记录定义代理行为模式和能力的系统提示。
    系统提示是代理personality和能力的核心定义。
    
    主要作用:
    - 定义代理的角色和能力范围
    - 设置行为规范和约束条件
    - 提供工具使用说明和示例
    - 建立与用户交互的基本模式
    
    属性:
        system_prompt (str): 系统提示的完整内容
    """
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        """
        转换为消息列表
        
        将系统提示转换为系统角色的消息。在摘要模式下，
        系统提示会被省略以减少上下文长度。
        
        参数:
            summary_mode (bool): 是否使用摘要模式
            
        返回:
            list[Message]: 包含系统提示的消息列表，摘要模式下返回空列表
        """
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    """
    最终答案步骤记录类 - 记录任务的最终完成结果
    
    该类用于标记任务执行的完成，记录代理产生的最终答案或输出。
    这是代理执行流程的终点，包含了任务的最终成果。
    
    主要作用:
    - 标记任务执行的结束
    - 记录最终的输出结果
    - 提供任务完成的明确信号
    - 支持结果的后续处理和分析
    
    属性:
        output (Any): 任务的最终输出，可以是任何类型的数据
    """
    output: Any


class AgentMemory:
    """
    代理记忆管理器 - 管理代理执行过程中的所有步骤和状态
    
    该类是代理记忆系统的核心组件，负责跟踪和存储代理的完整执行历史。
    它提供了一个统一的接口来管理不同类型的执行步骤，支持记忆的
    存储、检索、回放和分析等功能。
    
    核心功能:
    - 统一管理所有类型的执行步骤
    - 提供步骤的增删改查操作
    - 支持记忆的序列化和反序列化
    - 提供格式转换和消息生成功能
    - 支持调试和分析工具
    
    设计特点:
    - 类型安全的步骤管理
    - 灵活的序列化机制
    - 高效的内存使用
    - 可扩展的步骤类型支持
    
    属性:
        system_prompt (SystemPromptStep): 系统提示步骤，定义代理的基本行为
        steps (list): 所有执行步骤的有序列表，按执行顺序排列
    
    参数:
        system_prompt (str): 代理的系统提示内容，定义其行为模式和能力
    
    使用示例:
        memory = AgentMemory("你是一个有用的助手")
        memory.steps.append(TaskStep(task="计算 2+2"))
        memory.steps.append(ActionStep(step_number=1, timing=Timing(...)))
    """
    
    def __init__(self, system_prompt: str):
        """
        初始化代理记忆管理器
        
        创建一个新的记忆管理器实例，设置系统提示并初始化步骤列表。
        
        参数:
            system_prompt (str): 代理的系统提示内容
        """
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        """
        重置记忆状态
        
        清空所有已记录的执行步骤，但保留系统提示。
        通常在开始新的任务执行时调用。
        """
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """
        获取简洁的步骤列表
        
        返回简化版本的步骤列表，移除了模型输入消息等详细信息，
        主要用于快速概览和轻量级的数据传输。
        
        返回:
            list[dict]: 简化的步骤字典列表，不包含 model_input_messages 字段
        """
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """
        获取完整的步骤列表
        
        返回包含所有详细信息的完整步骤列表，用于详细分析、
        调试和完整的记忆重建。
        
        返回:
            list[dict]: 包含所有字段的完整步骤字典列表
        """
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """
        回放代理的执行步骤
        
        在控制台中以格式化的方式显示代理的完整执行历史，
        这是一个强大的调试和分析工具，帮助用户理解代理的
        决策过程和执行轨迹。
        
        回放功能特点:
        - 按时间顺序展示所有步骤
        - 使用丰富的视觉格式
        - 支持详细模式和简洁模式
        - 突出显示关键信息和错误
        
        参数:
            logger (AgentLogger): 用于打印回放日志的日志记录器，
                提供丰富的格式化输出功能
            detailed (bool, 可选): 如果为 True，还会显示每个步骤的
                详细记忆信息，包括模型输入消息等。默认为 False。
                注意：详细模式会指数级增加日志长度，仅推荐用于
                深度调试和问题诊断。
        
        使用场景:
        - 调试代理执行问题
        - 分析决策过程
        - 展示执行结果
        - 教学和演示
        """
        logger.console.log("Replaying the agent's steps:")
        
        # 显示系统提示
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        
        # 逐步回放所有执行步骤
        for step in self.steps:
            if isinstance(step, TaskStep):
                # 显示任务步骤
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                # 显示行动步骤
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                
                # 在详细模式下显示输入消息
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                
                # 显示模型输出
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                # 显示规划步骤
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                
                # 在详细模式下显示输入消息
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                
                # 显示规划内容
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)


__all__ = ["AgentMemory"]
