#!/usr/bin/env python3

__author__ = "xi"

import base64
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.sparse import csr_array


class SystemProfile(BaseModel):
    """Agent或其子模块自身的画像信息"""

    model_config = ConfigDict(extra="allow")

    description: str = Field(
        title="该模块的角色描述信息"
    )
    language: Optional[str] = Field(
        title="该模块所使用的语言",
        default=None
    )
    capabilities: Optional[List[str]] = Field(
        title="该模块的能力范畴（如能完成什么任务），可分条描述",
        default=None
    )
    constrains: Optional[List[str]] = Field(
        title="该模块的约束条件（如立场原则、特殊规则），可分条描述",
        default=None
    )


class ChatMessage(BaseModel):
    """对话消息"""

    model_config = ConfigDict(extra="allow")

    content: str = Field(
        title="消息内容"
    )
    role: Optional[str] = Field(
        title="发出该消息的角色，也可以不设定，将其包含在content中",
        default=None
    )
    thinking: Optional[str] = Field(
        title="产生该对话消息时的思考内容（如有）",
        default=None
    )


class Mention(BaseModel):
    """用户提及信息"""

    model_config = ConfigDict(extra="allow")

    content: str = Field(
        title="用户提及的内容，例如用户感兴趣的话题、实体等"
    )
    turn_id: Optional[Union[int, str]] = Field(
        title="对应提及信息的轮次标识，不一定式数字，可以是关于轮次的描述（如最近、很早以前）",
        default=None
    )


class SessionMemory(BaseModel):
    """会话级（Session Level）记忆"""

    model_config = ConfigDict(extra="allow")

    chat_history: List[ChatMessage] = Field(
        title="当前会话的历史对话信息，默认为空列表",
        default_factory=list
    )
    mentions: Optional[List[Mention]] = Field(
        title="当前会话中所有用户提及信息",
        default_factory=list
    )
    user_preference: Optional[Dict[str, str]] = Field(
        title="当前会话的用户偏好信息",
        default=None
    )


class UserMemory(BaseModel):
    """用户级（User Level）记忆"""

    model_config = ConfigDict(extra="allow")

    user_preference: Optional[Dict[str, str]] = Field(
        title="基于所有历史行为总结出的用户偏好信息",
        default=None
    )
    user_profile: Optional[Dict[str, str]] = Field(
        title="用户画像",
        default=None
    )


class SystemMemory(BaseModel):
    """系统级（System Level）记忆"""

    model_config = ConfigDict(extra="allow")

    few_shots: Optional[List[str]] = Field(
        title="针对当前任务的示例",
        default=None
    )
    domain_knowledge: Optional[List[str]] = Field(
        title="针对当前任务的领域知识",
        default=None
    )
    reflections: Optional[List[str]] = Field(
        title="针对当前任务的反思信息",
        default=None
    )


class ToolArgument(BaseModel):
    """工具属性（入参）描述"""

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = Field(
        title="该参数的数据类型",
        default=None
    )
    description: Optional[str] = Field(
        title="该参数的描述",
        default=None
    )
    properties: Optional[Dict[str, "ToolArgument"]] = None  # 支持嵌套对象
    items: Optional["ToolArgument"] = None  # 支持数组项的类型定义


class ToolSchema(BaseModel):
    """工具的输入输出格式描述"""

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        title="返回值类型",
        default="object"
    )
    arguments: Dict[str, ToolArgument] = Field(
        title="工具的入参描述",
        default_factory=dict
    )
    required: List[str] = Field(
        title="调用该工具必须给出的参数",
        default_factory=list
    )


class Tool(BaseModel):
    """工具描述"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="工具名称"
    )
    description: Optional[str] = Field(
        title="工具的功能描述",
        default=None
    )
    schema: Optional[ToolSchema] = Field(
        title="工具的输出输出格式信息",
        default=None
    )


class Intent(BaseModel):
    """候选意图描述"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="候选意图名称"
    )
    description: Optional[str] = Field(
        title="候选意图详细描述",
        default=None
    )


class IntentRequest(BaseModel):
    """意图理解请求"""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        title="用户原始输入"
    )
    candidate_intents: Optional[List[Intent]] = Field(
        title="候选意图",
        default=None
    )
    tools: Optional[List[Tool]] = Field(
        title="可使用的工具集合",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="意图理解模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="意图理解模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话级记忆信息",
        default=None
    )


class IntentResponse(BaseModel):
    """意图理解模块响应"""

    model_config = ConfigDict(extra="allow")

    intent: Intent = Field(
        title="基于用户输入解析出的用户意图信息"
    )
    candidate_tools: Optional[List[Tool]] = Field(
        title="基于当前用户意图筛选出的候选工具集合，若具体意图理解模块支持工具筛选可忽略该项",
        default=None
    )
    thinking: Optional[str] = Field(
        title="意图理解对应的思考信息（如有）",
        default=None
    )


class ToolCalling(BaseModel):
    """工具调用信息"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="被调用工具的名称"
    )
    arguments: Dict[str, Any] = Field(
        title="调用工具时，传递给工具的参数（key-value格式）",
        default_factory=dict
    )


class PlanningRequest(BaseModel):
    """单次任务规划请求"""

    model_config = ConfigDict(extra="allow")

    task: str = Field(
        title="任务描述，可以式用户query或改写后的query"
    )
    tools: List[Tool] = Field(
        title="完成该任务的候选工具",
        default_factory=list
    )
    intent: Optional[Intent] = Field(
        title="用户意图，可以由专用的意图理解模块传入，也可以由规划模块自行解析",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="任务规划模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="任务规划模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话级记忆信息",
        default=None
    )


class Plan(BaseModel):
    """任务规划结果"""

    model_config = ConfigDict(extra="allow")

    tool_callings: List[ToolCalling] = Field(
        title="完成相应任务的工具调用序列",
        default_factory=list
    )
    fallback: Optional[str] = Field(
        title="工具调用失败后返回的内容",
        default=None
    )
    thinking: Optional[str] = Field(
        title="生成该计划的思考信息（如有）",
        default=None
    )


class PlanningResponse(BaseModel):
    """任务规划响应"""

    model_config = ConfigDict(extra="allow")

    plans: Union[Plan, List[Plan]] = Field(
        title="完成该任务的规划，如返回多个Plan对象则表示其包含的子任务的规划",
        default=None
    )
    thinking: Optional[str] = Field(
        title="完成该任务的思考信息（如有）",
        default=None
    )


class ToolExecutingRequest(BaseModel):
    """工具执行模块请求"""

    model_config = ConfigDict(extra="allow")

    tool_callings: List[ToolCalling] = Field(
        title="完成相应任务的工具调用序列",
    )
    task: Optional[str] = Field(
        title="原始的任务描述",
        default=None
    )
    intent: Optional[Intent] = Field(
        title="用户意图",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="工具执行模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="工具执行模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话级记忆信息",
        default=None
    )


class ExecutionError(BaseModel):
    """执行错误信息对象"""

    model_config = ConfigDict(extra="allow")

    message: str = Field(
        title="错误信息"
    )
    error: Optional[str] = Field(
        title="错误（异常）名称",
        default=None
    )
    traceback: Optional[str] = Field(
        title="错误相关的详细信息",
        default=None
    )


class ExecutionStatus(BaseModel):
    """执行状态信息对象"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="被执行工具的名称"
    )
    result: Optional[Any] = Field(
        title="工具执行结果",
        default=None
    )
    error: Optional[ExecutionError] = Field(
        title="工具执行错误信息",
        default=None
    )


class ToolExecutingResponse(BaseModel):
    """工具执行模块响应对象"""

    model_config = ConfigDict(extra="allow")

    status: List[ExecutionStatus] = Field(
        title="各工具执行状态",
        default_factory=list
    )


class RewritingRequest(BaseModel):
    """改写模块请求"""

    model_config = ConfigDict(extra="allow")

    content: str = Field(
        title="原始内容"
    )
    system_profile: Optional[SystemProfile] = Field(
        title="改写模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="改写模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话级记忆信息",
        default=None
    )


class RewritingResponse(BaseModel):
    """改写模块响应"""

    model_config = ConfigDict(extra="allow")

    rewritten_content: str = Field(
        title="改写后的内容"
    )
    thinking: Optional[str] = Field(
        title="改写对应的思考信息（如有）",
        default=None
    )


class NDArray(BaseModel):
    """多维数组，可与numpy.ndarray相互转换"""

    data: str = Field(
        title="数组数据（base64编码的二进制数据）"
    )
    dtype: str = Field(
        title="数据类型"
    )
    shape: List[int] = Field(
        title="数据shape（支持高维数组）"
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
        title="数组数据（数据的列表）"
    )
    indices: List[int] = Field(
        title="索引序号"
    )
    indptr: List[int] = Field(
        title="行数据范围"
    )
    dtype: str = Field(
        title="数据类型"
    )
    shape: Tuple[int, int] = Field(
        title="数组shape（仅支持二维数组）"
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


class DenseEmbeddingRequest(BaseModel):
    """稠密向量表征请求"""

    model_config = ConfigDict(extra="allow")

    text: Union[str, List[str]] = Field(
        title="需要表征的文本"
    )
    normalize: bool = Field(
        title="是否对结果向量进行规一化",
        default=True
    )


class DenseEmbeddingResponse(BaseModel):
    """稠密向量表征响应"""

    model_config = ConfigDict(extra="allow")

    embedding: NDArray = Field(
        title="稠密表征向量（若表征多条文本，则返回矩阵）"
    )


class SparseEmbeddingRequest(BaseModel):
    """稀疏向量表征请求"""

    model_config = ConfigDict(extra="allow")

    text: Union[str, List[str]] = Field(
        title="需要表征的文本"
    )


class SparseEmbeddingResponse(BaseModel):
    """稀疏向量表征响应"""

    model_config = ConfigDict(extra="allow")

    embedding: CSRArray = Field(
        title="稀疏表征向量（若表征多条文本，则返回矩阵）"
    )


class RetrievalRequest(BaseModel):
    """知识库检索请求"""

    model_config = ConfigDict(extra="allow")

    collections: List[str] = Field(
        title="检索的集合，多个取值表示同时从多个集合中检索"
    )
    query: str = Field(
        title="检索Query"
    )
    top_k: int = Field(
        default=6,
        title="检索返回Top-K",
        ge=1,
        le=50
    )
    expr: Dict[str, str] = Field(
        default_factory=dict,
        title="元数据表达式"
    )
    index_fields: List[str] = Field(
        default=("vector",),
        title="索引字段的名称，默认为vector，可以指定多个表示混合检索"
    )
    output_fields: Optional[List[str]] = Field(
        default=None,
        title="检索返回Field字段列表, 默认返回除向量字段以外的所有字段"
    )
    use_rerank: bool = Field(
        default=False,
        title="是否使用Rerank"
    )
    pre_top_k: Optional[int] = Field(
        default=None,
        title="Rerank候选集大小",
        ge=1,
        le=50
    )


class RetrievalResponse(BaseModel):
    """知识库检索响应"""

    model_config = ConfigDict(extra="allow")

    distance: List[float] = Field(
        default_factory=list,
        title="检索距离"
    )
    scores: List[float] = Field(
        default_factory=list,
        title="重排分数"
    )
    items: List[Dict] = Field(
        default_factory=list,
        title="检索对象列表"
    )
