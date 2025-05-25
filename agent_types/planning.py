#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "PlanningRequest",
    "PlanningResponse",
]

from typing import List, Optional, Union

from pydantic import ConfigDict, Field

from agent_types.common import ExecutionStatus, Intent, Plan, Request, Response, SessionMemory, SystemMemory, \
    SystemProfile, Tool, UserMemory


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
    execution_status: Optional[List[ExecutionStatus]] = Field(
        title="执行状态",
        description="如果之前的Plan已经执行过，但存在错误，可以传给Planning参考以进行模块重新规划",
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
