#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "IntentRequest",
    "IntentResponse",
]

from typing import List, Optional

from pydantic import ConfigDict, Field

from agent_types.common import Intent, Request, Response, SessionMemory, SystemMemory, SystemProfile, Tool, UserMemory


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

    intents: List[Intent] = Field(
        title="意图",
        description="支持多个意图"
    )
    candidate_tools: Optional[List[List[Tool]]] = Field(
        title="候选工具集合",
        description="基于当前用户意图筛选出的候选工具集合，intents中的每个意图都对应各自的候选工具集",
        default=None
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="意图理解对应的思考信息（如有）",
        default=None
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
