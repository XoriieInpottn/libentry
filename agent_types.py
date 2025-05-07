#!/usr/bin/env python3

__author__ = "xi"

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SystemProfile(BaseModel):
    role: str
    language: str
    capabilities: str


class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: str


class Memory(BaseModel):
    user_profile: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = None


class SystemMemory(BaseModel):
    few_shots: Optional[List[str]] = None
    domain_knowledge: Optional[List[str]] = None


class ToolProperty(BaseModel):
    type: Optional[str]
    description: Optional[str] = None
    properties: Optional[Dict[str, "ToolProperty"]] = None  # 支持嵌套对象
    items: Optional["ToolProperty"] = None  # 支持数组项的类型定义


class ToolSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, ToolProperty] = {}
    required: List[str] = []


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    schema: Optional[ToolSchema] = None


class IntentRequest(BaseModel):
    query: str
    tools: List[Tool]
    memory: Optional[Memory] = None
    system_profile: Optional[SystemProfile] = None
    system_memory: Optional[SystemMemory] = None


class IntentResponse(BaseModel):
    intent: str
    candidate_tools: List[Tool]


class ToolCalling(BaseModel):
    name: str = Field(..., description="The name of the tool to be called.")
    arguments: Optional[Dict[str, Any]] = Field(
        ..., description="A dictionary of arguments to be passed to the tool."
    )


class PlanningRequest(BaseModel):
    task: str
    intent: str
    tools: List[Tool]
    memory: Optional[Memory] = None
    system_profile: Optional[SystemProfile] = None
    system_memory: Optional[SystemMemory] = None


class PlanPerTask(BaseModel):
    """Represents a plan for a specific task."""
    task: Optional[str] = Field(default=None, description="The task for which the plan is created")
    plan: Optional[Union[str, List[ToolCalling]]] = Field(
        default=None,
        description="The plan on how the agents can execute current task",
    )


class PlanningResponse(BaseModel):
    """Planning response"""
    list_of_plan_per_task: List[PlanPerTask] = Field(
        ...,
        description="The plans on how the agents can execute their tasks using the available tools",
    )
    text: str = Field(
        ...,
        description="The original textual output of the Planning",
    )


class ToolExecutingRequest(BaseModel):
    """Represents a request to execute a tool."""
    tool_call_list: List[ToolCalling] = Field(
        ...,
        description="The list of tools to execute",
    )
    memory: Optional[Memory] = Field(
        default=None,
        description="The memory"
    )
    query: Optional[str] = Field(
        default=None,
        description="The user query"
    )


class ExecutionInfo(BaseModel):
    success: bool = Field(..., description="whether successful")
    name: Optional[str] = Field(None, description="tool name")
    arguments: Optional[Dict[str, Any]] = Field(None, description="tool arguments")
    error_message: Optional[str] = Field(None, description="error message")
    error_code: Optional[int] = Field(None, description="error code")

    def __str__(self):
        status = "success" if self.success else f"failed (code: {self.error_code}, error info: {self.error_message})"
        return f"[{self.name or 'Unknown Tool'}] execute status: {status}"


class ToolExecutingResponse(BaseModel):
    """tool executing response."""
    result_list: List[Any] = Field(
        ...,
        description="The result of tool executing",
    )
    exec_info_list: List[ExecutionInfo] = Field(..., description="The execution info of tools")