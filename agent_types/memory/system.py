#!/usr/bin/env python3

__author__ = "xi"

from typing import List, Optional

from pydantic import ConfigDict, Field

from agent_types.common import FewShot, Request, Response, SessionMemory, SystemMemory, SystemProfile, UserMemory


class ReadFewShotsRequest(Request):
    """读取示例请求"""

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
    system_memory: Optional[SystemMemory] = Field(
        title="当前系统记忆对象",
        description="若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadFewShotsResponse(Response):
    """读取示例响应"""

    model_config = ConfigDict(extra="allow")

    few_shots: List[FewShot] = Field(
        title="示例信息",
        description="读取到的示例信息"
    )
    system_memory: Optional[SystemMemory] = Field(
        title="输出系统记忆对象",
        description="只有请求时指定了当前记忆，这里才会输出",
        default=None
    )


class WriteFewShotsRequest(Request):
    """写入示例请求"""

    model_config = ConfigDict(extra="allow")

    collection: str = Field(
        title="集合名称",
        description="用于存储写入示例的集合名称"
    )
    few_shots: List[FewShot] = Field(
        title="示例内容",
        description="写入的示例内容"
    )


class WriteFewShotsResponse(Response):
    """示例写入响应"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class ReadDomainKnowledgeRequest(Request):
    """读取领域知识请求"""

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
    system_memory: Optional[SystemMemory] = Field(
        title="当前系统记忆对象",
        description="若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadDomainKnowledgeResponse(Response):
    """读取领域知识响应"""

    model_config = ConfigDict(extra="allow")

    domain_knowledge: List[str] = Field(
        title="领域知识",
        description="读取到的领域知识信息"
    )
    system_memory: Optional[SystemMemory] = Field(
        title="输出系统记忆对象",
        description="只有请求时指定了当前记忆，这里才会输出",
        default=None
    )


class WriteDomainKnowledgeRequest(Request):
    """写入领域知识请求"""

    model_config = ConfigDict(extra="allow")

    collection: str = Field(
        title="集合名称",
        description="用于存储写入示例的集合名称"
    )
    domain_knowledge: List[str] = Field(
        title="领域知识内容",
        description="写入的领域知识内容"
    )


class WriteDomainKnowledgeResponse(Response):
    """写入领域知识响应"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class ReadReflectionsRequest(Request):
    """读取反思内容请求"""

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
    system_memory: Optional[SystemMemory] = Field(
        title="当前系统记忆对象",
        description="若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadReflectionsResponse(Response):
    """读取反思内容响应"""

    model_config = ConfigDict(extra="allow")

    reflections: List[str] = Field(
        title="反思内容",
        description="读取到的反思内容信息"
    )
    system_memory: Optional[SystemMemory] = Field(
        title="输出系统记忆对象",
        description="只有请求时指定了当前记忆，这里才会输出",
        default=None
    )


class WriteReflectionsRequest(Request):
    """写入反思内容请求"""

    model_config = ConfigDict(extra="allow")

    collection: str = Field(
        title="集合名称",
        description="用于存储写入示例的集合名称"
    )
    reflections: List[str] = Field(
        title="反思内容",
        description="写入的反思内容"
    )


class WriteReflectionsResponse(Response):
    """写入反思内容响应"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class ExtractReflectionRequest(Request):
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


class ExtractReflectionResponse(Response):
    """反思响应对象"""

    model_config = ConfigDict(extra="allow")

    reflection: Optional[str] = Field(
        title="反思结果",
        description="对应反思的结果，如果请求时指定了集合名称，则可以不返回该项",
        default=None
    )
