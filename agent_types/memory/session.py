#!/usr/bin/env python3

__author__ = "xi"

from typing import Dict, List, Optional, Union

from pydantic import ConfigDict, Field

from agent_types.common import ChatMessage, Mention, Request, Response, SessionMemory


class ReadChatHistoryRequest(Request):
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
    session_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ChatHistoryReadingResponse(Response):
    """对话历史读取响应"""

    model_config = ConfigDict(extra="allow")

    chat_history: List[ChatMessage] = Field(
        title="对话历史",
        description="读取到的对话历史信息"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class ReadMentionsRequest(Request):
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
    session_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadMentionsResponse(Response):
    """提及信息读取响应"""

    model_config = ConfigDict(extra="allow")

    mentions: List[Mention] = Field(
        title="用户提及信息",
        description="读取到的用户提及信息"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class ReadSessionPreferenceRequest(Request):
    """当前会话的用户偏好读取请求"""

    model_config = ConfigDict(extra="allow")

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadSessionPreferenceResponse(Response):
    """前会话的用户偏好读取响应"""

    model_config = ConfigDict(extra="allow")

    session_preference: Dict[str, str] = Field(
        title="会话用户偏好",
        description="读取到的会话用户偏好信息"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )
