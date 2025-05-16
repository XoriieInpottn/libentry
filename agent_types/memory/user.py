#!/usr/bin/env python3

__author__ = "xi"

from typing import Dict, Optional, Union

from pydantic import ConfigDict, Field

from agent_types.common import Request, Response, SessionMemory, SystemMemory, SystemProfile, UserMemory


class ReadUserMemoryRequest(Request):
    """一次性读取用户记忆"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    user_memory: Optional[UserMemory] = Field(
        title="当前用户记忆对象",
        description="若给定，则该内容会被整合到响应对象中",
        default=None
    )


class ReadUserMemoryResponse(Response):
    """一次性读取用户记忆"""

    model_config = ConfigDict(extra="allow")

    user_memory: UserMemory = Field(
        title="读取后的用户记忆对象",
        description="无论在请求读取时候是否给定当前记忆，都会返回"
    )


class ReadUserPreferenceRequest(Request):
    """读取用户偏好"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    user_memory: Optional[UserMemory] = Field(
        title="当前用户记忆对象",
        description="若给定，则该内容会被整合到响应对象中",
        default=None
    )


class ReadUserPreferenceResponse(Response):
    """读取用户偏好"""

    model_config = ConfigDict(extra="allow")

    user_preference: Dict[str, str] = Field(
        title="用户偏好",
        description="读取到的用户偏好信息"
    )
    user_memory: Optional[UserMemory] = Field(
        title="读取后的用户记忆对象",
        description="若在请求中给定了当前记忆，则这里返回的是整合后的记忆对象",
        default=None
    )


class ReadUserProfileRequest(Request):
    """读取用户画像"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    user_memory: Optional[UserMemory] = Field(
        title="当前用户记忆对象",
        description="若给定，则该内容会被整合到响应对象中",
        default=None
    )


class ReadUserProfileResponse(Response):
    """用户画像读取响应"""

    model_config = ConfigDict(extra="allow")

    user_profile: Dict[str, str] = Field(
        title="用户画像",
        description="读取到的用户画像信息"
    )
    user_memory: Optional[UserMemory] = Field(
        title="读取后的用户记忆对象",
        description="若在请求中给定了当前记忆，则这里返回的是整合后的记忆对象",
        default=None
    )


class WriteUserPreferenceRequest(Request):
    """写入用户偏好"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    user_preference: Dict[str, str] = Field(
        title="用户偏好",
        description="要写入的用户偏好，需要在调用写入之前先进行提取",
    )


class WriteUserPreferenceResponse(Response):
    """写入用户偏好"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class WriteUserProfileRequest(Request):
    """写入用户画像"""

    model_config = ConfigDict(extra="allow")

    user_id: Union[str, int] = Field(
        title="用户标识",
        description="用户标识"
    )
    user_profile: Dict[str, str] = Field(
        title="用户画像",
        description="要写入的用户画像，需要在调用写入之前先进行提取"
    )


class WriteUserProfileResponse(Response):
    """写入用户画像"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class ExtractUserPreferenceRequest(Request):
    """用户偏好提取请求"""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        title="用户查询",
        description="用于分析用户偏好的查询内容"
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


class ExtractUserPreferenceResponse(Response):
    """用户偏好提取响应"""

    model_config = ConfigDict(extra="allow")

    user_preference: Dict[str, str] = Field(
        title="用户偏好信息",
        description="用户偏好信息，如果在请求时给定了collection，则可以不返回",
        default=None
    )


class ExtractUserProfileRequest(Request):
    """用户画像提取请求"""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        title="用户查询",
        description="用于分析用户画像的查询内容"
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


class ExtractUserProfileResponse(Response):
    """用户画像提取响应"""

    model_config = ConfigDict(extra="allow")

    user_profile: Optional[Dict[str, str]] = Field(
        title="用户画像信息",
        description="用户画像信息，如果在请求时给定了collection，则可以不返回",
        default=None
    )
