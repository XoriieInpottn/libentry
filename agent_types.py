#!/usr/bin/env python3

__author__ = "xi"

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


class FewShotsReadingRequest(Request):
    """示例读取请求"""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        title="用户查询",
        description="检索示例所需的用户查询内容"
    )
    collection: str = Field(
        title="集合名称",
        description="检索示例的集合名称"
    )
    n: int = Field(
        title="读取最大条数",
        description="读取最大条数"
    )
    source_memory: Optional[SystemMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class FewShotsReadingResponse(Response):
    """示例读取响应"""

    model_config = ConfigDict(extra="allow")

    few_shots: List[FewShot] = Field(
        title="示例信息",
        description="读取到的示例信息"
    )
    output_memory: Optional[SystemMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了输入记忆，这里才会输出",
        default=None
    )


class FewShotsWritingRequest(Request):
    """示例写入请求对象"""

    model_config = ConfigDict(extra="allow")

    collection: str = Field(
        title="集合名称",
        description="用于存储写入示例的集合名称"
    )
    few_shots: Optional[List[FewShot]] = Field(
        title="示例内容",
        description="写入的示例内容",
        default=None
    )
    source_memory: Optional[SystemMemory] = Field(
        title="源记忆对象",
        description="写入内容所在的源记忆对象，“示例内容”和“源记忆对象”至少应该给出一个",
        default=None
    )

    def model_post_init(self, _):
        if self.few_shots is None and self.source_memory is None:
            raise ValueError("At least one of {few_shots, source_memory} should be given.")


class FewShotsWritingResponse(Response):
    """示例写入响应对象"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class DomainKnowledgeReadingRequest(Request):
    """领域知识读取请求"""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        title="用户查询",
        description="用于检索领域知识的用户查询内容"
    )
    collection: str = Field(
        title="集合名称",
        description="用于检索领域知识的集合名称"
    )
    n: int = Field(
        title="读取最大条数",
        description="读取领域知识的最大条数"
    )
    source_memory: Optional[SystemMemory] = Field(
        title="源记忆对象",
        description="若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class DomainKnowledgeReadingResponse(Response):
    """领域知识读取响应"""

    model_config = ConfigDict(extra="allow")

    domain_knowledge: List[str] = Field(
        title="领域知识",
        description="读取到的领域知识信息"
    )
    output_memory: Optional[SystemMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class ReflectionsReadingRequest(Request):
    """反思信息读取请求"""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        title="用户查询",
        description="用于检索反思信息的用户查询"
    )
    collection: str = Field(
        title="集合名称",
        description="用于检索反思信息的集合名称"
    )
    n: int = Field(
        title="读取最大条数",
        description="读取反思信息的最大条数"
    )
    source_memory: Optional[SystemMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReflectionsReadingResponse(Response):
    """反思信息读取响应"""

    model_config = ConfigDict(extra="allow")

    reflections: List[str] = Field(
        title="反思信息",
        description="读取到的反思信息"
    )
    output_memory: Optional[SystemMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
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
        title="用户户思想",
        description="基于所有历史行为总结出的用户画像信息",
        default=None
    )


class UserPreferenceReadingRequest(Request):
    """用户偏好读取请求"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    source_memory: Optional[UserMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class UserPreferenceReadingResponse(Response):
    """用户偏好读取响应"""

    model_config = ConfigDict(extra="allow")

    user_preference: Dict[str, str] = Field(
        title="用户偏好",
        description="读取到的用户偏好信息"
    )
    output_memory: Optional[UserMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class UserProfileReadingRequest(Request):
    """用户画像读取请求"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    source_memory: Optional[UserMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class UserProfileReadingResponse(Response):
    """用户画像读取响应"""

    model_config = ConfigDict(extra="allow")

    user_profile: Dict[str, str] = Field(
        title="用户画像",
        description="读取到的用户画像信息"
    )
    output_memory: Optional[UserMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
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


class ChatHistoryReadingRequest(Request):
    """对话历史读取请求"""

    model_config = ConfigDict(extra="allow")

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    n: int = Field(
        title="最大对话轮数",
        description="最大对话轮数"
    )
    source_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ChatHistoryReadingResponse(Response):
    """对话历史读取响应"""

    model_config = ConfigDict(extra="allow")

    chat_messages: List[ChatMessage] = Field(
        title="对话历史",
        description="读取到的对话历史信息"
    )
    output_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class MentionsReadingRequest(Request):
    """提及信息读取请求"""

    model_config = ConfigDict(extra="allow")

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    n: int = Field(
        title="最大信息条数",
        description="最大信息条数"
    )
    source_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class MentionsReadingResponse(Response):
    """提及信息读取响应"""

    model_config = ConfigDict(extra="allow")

    mentions: List[Mention] = Field(
        title="用户提及信息",
        description="读取到的用户提及信息"
    )
    output_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class SessionPreferenceReadingRequest(Request):
    """当前会话的用户偏好读取请求"""

    model_config = ConfigDict(extra="allow")

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    n: int = Field(
        title="最大信息条数",
        description="最大信息条数"
    )
    source_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class SessionPreferenceReadingResponse(Response):
    """前会话的用户偏好读取响应"""

    model_config = ConfigDict(extra="allow")

    session_preference: Dict[str, str] = Field(
        title="会话用户偏好",
        description="读取到的会话用户偏好信息"
    )
    output_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
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


class DenseEmbeddingRequest(Request):
    """稠密向量表征请求"""

    model_config = ConfigDict(extra="allow")

    text: Union[str, List[str]] = Field(
        title="文本",
        description="需要表征的文本"
    )
    normalize: bool = Field(
        title="是否规一化",
        description="是否对结果向量进行规一化",
        default=True
    )


class DenseEmbeddingResponse(Response):
    """稠密向量表征响应"""

    model_config = ConfigDict(extra="allow")

    embedding: NDArray = Field(
        title="表征向量",
        description="稠密表征向量（若表征多条文本，则返回矩阵）"
    )


class SparseEmbeddingRequest(Request):
    """稀疏向量表征请求"""

    model_config = ConfigDict(extra="allow")

    text: Union[str, List[str]] = Field(
        title="文本",
        description="需要表征的文本"
    )


class SparseEmbeddingResponse(Response):
    """稀疏向量表征响应"""

    model_config = ConfigDict(extra="allow")

    embedding: CSRArray = Field(
        title="表征向量",
        description="稀疏表征向量（若表征多条文本，则返回矩阵）"
    )


class RetrievalRequest(Request):
    """知识库检索请求"""

    model_config = ConfigDict(extra="allow")

    collections: List[str] = Field(
        title="检索集合名称",
        description="检索集合名称，多个取值表示同时从多个集合中检索"
    )
    query: str = Field(
        title="用户查询",
        description="检索所需的用户查询"
    )
    top_k: int = Field(
        title="检索返回Top-k",
        description="检索返回Top-k",
        default=6,
        ge=1,
        le=50
    )
    expr: Dict[str, str] = Field(
        title="元数据表达式",
        description="元数据表达式，即筛选条件",
        default_factory=dict
    )
    index_fields: List[str] = Field(
        title="索引字段名称",
        description="索引字段的名称，默认为vector，可以指定多个表示混合检索",
        default=("vector",)
    )
    output_fields: Optional[List[str]] = Field(
        title="返回字段",
        description="检索结果中需包含的字段列表, 默认返回除向量字段以外的所有字段",
        default=None
    )
    use_rerank: bool = Field(
        title="是否使用Rerank",
        description="是否使用Rerank",
        default=False
    )
    pre_top_k: Optional[int] = Field(
        title="Rerank候选集大小",
        description="Rerank候选集大小",
        default=None,
        ge=1,
        le=50
    )


class RetrievalResponse(Response):
    """知识库检索响应"""

    model_config = ConfigDict(extra="allow")

    distance: List[float] = Field(
        title="检索距离",
        description="检索距离",
        default_factory=list
    )
    scores: List[float] = Field(
        title="重排分数",
        description="重排分数",
        default_factory=list
    )
    items: List[Dict] = Field(
        title="检索对象列表",
        description="检索对象列表",
        default_factory=list
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


class IntentRequest(Request):
    """意图理解请求"""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        title="用户查询",
        description="用户原始输入的查询内容"
    )
    candidate_intents: Optional[List[Intent]] = Field(
        title="候选意图",
        description="候选意图，用于限定系统中用户意图的范围",
        default=None
    )
    tools: Optional[List[Tool]] = Field(
        title="可用工具集合",
        description="可使用的工具集合，用于根据用户意图进行工具的筛选，以减小规划模块的复杂度",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="意图理解模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="意图理解模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户记忆对象",
        description="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话记忆对象",
        description="会话级记忆信息",
        default=None
    )


class IntentResponse(Response):
    """意图理解模块响应"""

    model_config = ConfigDict(extra="allow")

    intent: Intent = Field(
        title="意图",
        description="基于用户输入解析出的用户意图信息"
    )
    candidate_tools: Optional[List[Tool]] = Field(
        title="候选工具集合",
        description="基于当前用户意图筛选出的候选工具集合，若具体意图理解模块支持工具筛选可忽略该项",
        default=None
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="意图理解对应的思考信息（如有）",
        default=None
    )


class PlanningRequest(Request):
    """单次任务规划请求"""

    model_config = ConfigDict(extra="allow")

    task: str = Field(
        title="任务描述",
        description="任务描述，可以式用户query或改写后的query"
    )
    tools: List[Tool] = Field(
        title="候选工具集合",
        description="完成该任务的候选工具，可以是所有工具，也可以是筛选后的",
        default_factory=list
    )
    intent: Optional[Intent] = Field(
        title="用户意图",
        description="用户意图，可以由专用的意图理解模块传入，也可以由规划模块自行解析",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="任务规划模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="任务规划模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户记忆对象",
        description="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话记忆对象",
        description="会话级记忆信息",
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


class PlanningResponse(Response):
    """任务规划响应"""

    model_config = ConfigDict(extra="allow")

    plans: Union[Plan, List[Plan]] = Field(
        title="规划方案",
        description="完成该任务的规划，如返回多个Plan对象则表示其包含的子任务的规划",
        default=None
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="完成该任务的思考信息（如有）",
        default=None
    )


class ToolExecutingRequest(Request):
    """工具执行模块请求"""

    model_config = ConfigDict(extra="allow")

    tool_callings: List[ToolCalling] = Field(
        title="工具调用序列",
        description="完成相应任务的工具调用序列",
    )
    task: Optional[str] = Field(
        title="原始任务描述",
        description="原始任务描述，可用于工具参数错误的时候对其进行纠正",
        default=None
    )
    intent: Optional[Intent] = Field(
        title="用户意图",
        description="用户意图",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="工具执行模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="工具执行模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户记忆对象",
        description="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话记忆对象",
        description="会话级记忆信息",
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


class ToolExecutingResponse(Response):
    """工具执行模块响应对象"""

    model_config = ConfigDict(extra="allow")

    status: List[ExecutionStatus] = Field(
        title="执行状态",
        description="各工具执行状态，可以用最后一个状态表示最终执行状态",
        default_factory=list
    )


class RewritingRequest(Request):
    """改写模块请求"""

    model_config = ConfigDict(extra="allow")

    content: str = Field(
        title="原始内容",
        description="原始内容"
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="改写模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="改写模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户记忆对象",
        description="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话记忆对象",
        description="会话级记忆信息",
        default=None
    )


class RewritingResponse(Response):
    """改写模块响应"""

    model_config = ConfigDict(extra="allow")

    rewritten_content: str = Field(
        title="改写后的内容",
        description="改写后的内容"
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="改写对应的思考信息（如有）",
        default=None
    )


class ReflectionRequest(Request):
    """反思请求对象"""

    model_config = ConfigDict(extra="allow")

    collection: str = Field(
        title="写入集合名称",
        description="反思的结果写入的集合"
    )
    input: str = Field(
        title="输入",
        description="反思对应的输入"
    )
    output: str = Field(
        title="输出",
        description="反思对应的输出"
    )
    feedback: str = Field(
        title="反馈",
        description="针对该输出输出的反馈信息"
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="得到对应输出的思考过程（如有）",
        default=None
    )
    tool_usage: Optional[str] = Field(
        title="工具调用",
        description="得到对应输出的工具使用情况（如有）",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="反思模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="反思模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户记忆对象",
        description="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话记忆对象",
        description="会话级记忆信息",
        default=None
    )


class ReflectionResponse(Response):
    """反思响应对象"""

    model_config = ConfigDict(extra="allow")

    reflection: Optional[str] = Field(
        title="反思结果",
        description="对应反思的结果，如果请求时指定了集合名称，则可以不返回该项",
        default=None
    )


class UserPreferenceExtractingRequest(Request):
    """用户偏好提取请求"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    query: str = Field(
        title="用户查询",
        description="用于分析用户偏好的查询内容"
    )
    collection: Optional[str] = Field(
        title="用户偏好存储集合",
        description="用户偏好存储集合，为空则表示不进行存储，而是将提取结果在Response中返回",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="用户偏好提取模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="用户偏好提取模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户记忆对象",
        description="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话记忆对象",
        description="会话级记忆信息",
        default=None
    )


class UserPreferenceExtractingResponse(Response):
    """用户偏好提取响应"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    user_preference: Optional[Dict[str, str]] = Field(
        title="用户偏好信息",
        description="用户偏好信息，如果在请求时给定了collection，则可以不返回",
        default=None
    )


class UserProfileExtractingRequest(Request):
    """用户画像提取请求"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    query: str = Field(
        title="用户查询",
        description="用于分析用户画像的查询内容"
    )
    collection: Optional[str] = Field(
        title="用户画像存储集合",
        description="用户画像存储集合，为空则表示不进行存储，而是将提取结果在Response中返回",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="用户画像提取模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="用户画像提取模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户记忆对象",
        description="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话记忆对象",
        description="会话级记忆信息",
        default=None
    )


class UserProfileExtractingResponse(Response):
    """用户画像提取响应"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    user_profile: Optional[Dict[str, str]] = Field(
        title="用户画像信息",
        description="用户画像信息，如果在请求时给定了collection，则可以不返回",
        default=None
    )
