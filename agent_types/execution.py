#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ToolExecutingRequest",
    "ToolExecutingResponse",
]

from typing import List, Optional

from pydantic import ConfigDict, Field

from agent_types.common import ExecutionStatus, Intent, Request, Response, SessionMemory, SystemMemory, SystemProfile, \
    ToolCalling, UserMemory


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


class ToolExecutingResponse(Response):
    """工具执行模块响应对象"""

    model_config = ConfigDict(extra="allow")

    status: List[ExecutionStatus] = Field(
        title="执行状态",
        description="各工具执行状态，可以用最后一个状态表示最终执行状态",
        default_factory=list
    )
