#!/usr/bin/env python3

__author__ = "xi"

from typing import Dict, List, Optional

from pydantic import BaseModel


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
    description: Optional[str]


class ToolSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, ToolProperty] = {}
    required: List[str] = []


class Tool(BaseModel):
    name: str
    description: Optional[str]
    schema: Optional[ToolSchema] = None


class PlanningRequest(BaseModel):
    task: str
    intent: str
    tools: List[Tool]
    system_profile: Optional[SystemProfile] = None
    system_memory: Optional[SystemMemory] = None


class PlanningResponse(BaseModel):
    pass
