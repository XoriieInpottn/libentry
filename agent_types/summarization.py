#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "SummarizationRequest",
    "SummarizationResponse",
]

from typing import Optional

from pydantic import ConfigDict, Field

from agent_types.common import GenerationOptions, Intent, Request, Response, SessionMemory, SystemMemory, SystemProfile, \
    UserMemory


class SummarizationRequest(Request):
    """总结模块
    主要依靠LLM实现，LLM的Prompt信息主要涵盖在system_profile和system_memory中了
    一个具体的总结模块服务就对应一个具体的LLM
    如果需要使用多个LLM进行不同用途的总结，请启动多个summarization服务，将LLM相关的配置信息在服务的启动项中给出
    """

    model_config = ConfigDict(extra="allow")

    task: str = Field(
        title="任务描述",
        description="任务描述，可以式用户query或改写后的query"
    )
    intent: Optional[Intent] = Field(
        title="用户意图",
        description="用户意图，可以由专用的意图理解模块传入，也可以由规划模块自行解析",
        default=None
    )
    generation_options: Optional[GenerationOptions] = Field(
        title="生成选项",
        description="可以不指定，如果不指定则以具体实现的默认值为准",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="总结模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="总结模块对应的系统级记忆信息",
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


class SummarizationResponse(Response):
    """总结模块
    如果是流式输出，则返回 Generator[SummarizationResponse, None, Optional[SummarizationResponse]]，
    或者输出 Iterable[SummarizationResponse]
    """

    model_config = ConfigDict(extra="allow")

    content: str = Field(
        title="总结内容",
        description="总结内容"
    )
    num_tokens: Optional[int] = Field(
        title="Token数量",
        description="产生这些总结内容所生成的Token数量",
        default=None
    )
