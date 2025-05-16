#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "Request",
    "Response",
    "SystemProfile",
    "FewShot",
    "SystemMemory",
    "UserMemory",
    "ChatMessage",
    "Mention",
    "SessionMemory",
    "NDArray",
    "CSRArray",
    "Property",
    "Tool",
    "Intent",
    "ToolCalling",
    "Plan",
    "ExecutionError",
    "ExecutionStatus",
]

import base64
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.sparse import csr_array


class Request(BaseModel):
    """所有请求对象的基类"""

    model_config = ConfigDict(extra="allow")

    trace_id: str = Field(
        title="该次请求的标识符",
        description="该次请求的标识符，主要用户跟踪该次请求的日志信息，如果不指定会自动生成",
        default_factory=lambda: str(uuid.uuid4())
    )


class Response(BaseModel):
    """所有响应对象的基类"""

    model_config = ConfigDict(extra="allow")

    trace_id: str = Field(
        title="被响应请求的标识符",
        description="该次请求的标识符，主要用户跟踪该次请求的日志信息，如果不指定会自动生成",
        default_factory=lambda: str(uuid.uuid4())
    )


class SystemProfile(BaseModel):
    """Agent或其子模块自身的画像信息"""

    model_config = ConfigDict(extra="allow")

    description: str = Field(
        title="角色描述信息",
        description="角色描述信息，用于为通用模块指定角色信息，描述功能作用"
    )
    language: Optional[str] = Field(
        title="语言",
        description="模块所使用的语言",
        default=None
    )
    capabilities: Optional[List[str]] = Field(
        title="能力范畴",
        description="该模块的能力范畴（如能完成什么任务），可分条描述",
        default=None
    )
    constrains: Optional[List[str]] = Field(
        title="条件约束",
        description="该模块的约束条件（如立场原则、特殊规则），可分条描述",
        default=None
    )


class FewShot(BaseModel):
    """示例对象"""

    model_config = ConfigDict(extra="allow")

    input: str = Field(
        title="输入内容",
        description="示例对应的输入内容"
    )
    output: str = Field(
        title="输入内容",
        description="示例对应的输出内容"
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="得到对应输出的思考过程（如有）",
        default=None
    )


class SystemMemory(BaseModel):
    """系统级（System Level）记忆"""

    model_config = ConfigDict(extra="allow")

    few_shots: Optional[List[FewShot]] = Field(
        title="示例",
        description="针对当前任务的示例，可以是Agent设计者预先给出",
        default=None
    )
    domain_knowledge: Optional[List[str]] = Field(
        title="领域知识",
        description="针对当前任务的领域知识，主要由于Agent设计者预先给出",
        default=None
    )
    reflections: Optional[List[str]] = Field(
        title="反思信息",
        description="针对当前任务的反思信息，主要由反思模块根据Agent历史表现总结得到",
        default=None
    )


class UserMemory(BaseModel):
    """用户级（User Level）记忆"""

    model_config = ConfigDict(extra="allow")

    user_preference: Optional[Dict[str, str]] = Field(
        title="用户偏好",
        description="基于所有历史行为总结出的用户偏好信息",
        default=None
    )
    user_profile: Optional[Dict[str, str]] = Field(
        title="用户画像",
        description="基于所有历史行为总结出的用户画像信息",
        default=None
    )


class ChatMessage(BaseModel):
    """对话消息"""

    model_config = ConfigDict(extra="allow")

    content: str = Field(
        title="消息内容",
        description="消息内容"
    )
    role: Optional[str] = Field(
        title="角色",
        description="发出该消息的角色，也可以不设定，将其包含在content中",
        default=None
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="产生该对话消息时的思考内容（如有）",
        default=None
    )


class Mention(BaseModel):
    """用户提及信息"""

    model_config = ConfigDict(extra="allow")

    content: str = Field(
        title="提及内容",
        description="用户提及的内容，例如用户感兴趣的话题、实体等"
    )
    turn_id: Optional[Union[int, str]] = Field(
        title="轮次标识",
        description="对应提及信息的轮次标识，不一定式数字，可以是关于轮次的描述（如最近、很早以前）",
        default=None
    )


class SessionMemory(BaseModel):
    """会话级（Session Level）记忆"""

    model_config = ConfigDict(extra="allow")

    chat_history: List[ChatMessage] = Field(
        title="对户历史",
        description="当前会话的历史对话信息，默认为空列表",
        default_factory=list
    )
    mentions: Optional[List[Mention]] = Field(
        title="用户提及信息",
        description="当前会话中所有用户提及信息",
        default_factory=list
    )
    session_preference: Optional[Dict[str, str]] = Field(
        title="用户偏好",
        description="当前会话的用户偏好信息",
        default=None
    )


class NDArray(BaseModel):
    """多维数组，可与numpy.ndarray相互转换"""

    data: str = Field(
        title="数组数据",
        description="数组数据（base64编码的二进制数据）"
    )
    dtype: str = Field(
        title="数据类型",
        description="数据类型"
    )
    shape: List[int] = Field(
        title="数组shape",
        description="数组shape（支持高维数组）"
    )

    @classmethod
    def from_array(cls, a: np.ndarray):
        bin_data = a.tobytes("C")
        return cls(
            data=base64.b64encode(bin_data).decode("utf-8"),
            dtype=str(a.dtype),
            shape=a.shape
        )

    def to_array(self) -> np.ndarray:
        bin_data = base64.b64decode(self.data)
        return np.frombuffer(
            buffer=bin_data,
            dtype=self.dtype
        ).reshape(self.shape)


class CSRArray(BaseModel):
    """稀疏数组，可与scipy.sparse.csr_array相互转换"""

    data: List[Union[int, float]] = Field(
        title="数组数据",
        description="数组数据（数据的列表）"
    )
    indices: List[int] = Field(
        title="索引序号",
        description="索引序号"
    )
    indptr: List[int] = Field(
        title="行数据范围",
        description="行数据范围"
    )
    dtype: str = Field(
        title="数据类型",
        description="数据类型"
    )
    shape: Tuple[int, int] = Field(
        title="数组shape",
        description="数组shape（仅支持二维数组）"
    )

    @classmethod
    def from_array(cls, a: csr_array):
        return cls(
            data=a.data,
            indices=a.indices,
            indptr=a.indptr,
            dtype=a.dtype,
            shape=a.shape
        )

    def to_array(self) -> csr_array:
        return csr_array(
            (self.data, self.indices, self.indptr),
            dtype=self.dtype,
            shape=self.shape
        )


class Property(BaseModel):
    """工具属性信息"""

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        title="数据类型",
        description="该属性的数据类型，通常为object, array, string, integer, float, bool或null",
        default="object"
    )
    description: Optional[str] = Field(
        title="属性描述",
        description="属性描述",
        default=None
    )
    properties: Optional[Dict[str, "Property"]] = Field(
        title="子属性信息",
        description="如果属性类型为object，该项为其子属性信息",
        default=None
    )
    required: Optional[List[str]] = Field(
        title="子属性中的必要属性",
        description="子属性中的必要属性",
        default=None
    )
    items: Optional["Property"] = Field(
        title="数组元素属性信息",
        description="如果属性类型为array，该项为其元素的属性信息",
        default=None
    )


class Tool(BaseModel):
    """工具描述"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="工具名称",
        description="工具名称"
    )
    description: Optional[str] = Field(
        title="工具描述",
        description="工具的功能描述",
        default=None
    )
    input_schema: Optional[Property] = Field(
        title="工具参数信息",
        description="工具的参数信息",
        default=None
    )


class Intent(BaseModel):
    """候选意图描述"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="意图名称",
        description="意图名称，可以用一个词或短语表示，也可以是一句话"
    )
    description: Optional[str] = Field(
        title="意图描述",
        description="意图详细描述",
        default=None
    )


class ToolCalling(BaseModel):
    """工具调用信息"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="工具名称",
        description="被调用工具的名称"
    )
    arguments: Dict[str, Any] = Field(
        title="调用参数",
        description="调用工具时，传递给工具的参数（key-value格式）",
        default_factory=dict
    )


class Plan(BaseModel):
    """任务规划结果"""

    model_config = ConfigDict(extra="allow")

    tool_callings: List[ToolCalling] = Field(
        title="工具调用序列",
        description="完成相应任务的工具调用序列",
        default_factory=list
    )
    fallback: Optional[str] = Field(
        title="兜底信息",
        description="工具调用失败后返回的内容",
        default=None
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="生成该计划的思考信息（如有）",
        default=None
    )


class ExecutionError(BaseModel):
    """执行错误信息对象"""

    model_config = ConfigDict(extra="allow")

    message: str = Field(
        title="错误信息",
        description="错误信息"
    )
    error: Optional[str] = Field(
        title="错误名称",
        description="通常是系统中捕获的异常的名称",
        default=None
    )
    traceback: Optional[str] = Field(
        title="详细信息",
        description="错误相关的详细信息，如函数调用栈信息等，用于定位错误",
        default=None
    )


class ExecutionStatus(BaseModel):
    """执行状态信息对象"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="工具名称",
        description="被执行工具的名称"
    )
    result: Optional[Any] = Field(
        title="执行结果",
        description="工具执行结果",
        default=None
    )
    error: Optional[ExecutionError] = Field(
        title="错误信息",
        description="工具执行错误信息，如果错误信息不为空，则执行结果无效",
        default=None
    )
