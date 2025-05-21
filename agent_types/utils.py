#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "NERRequest",
    "NERResponse",
    "ItemMatchRequest",
    "ItemMatchResponse",
]

from typing import List, Optional

from pydantic import ConfigDict, Field

from agent_types.common import Request, Response


class NERRequest(Request):
    """实体抽取请求"""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        title="用户查询",
        description="原始用户查询",
    )


class NERResponse(Response):
    """实体抽取响应"""

    model_config = ConfigDict(extra="allow")

    entities: List[str] = Field(
        title="实体列表",
        description="如果不存在实体则为空列表",
        default_factory=list
    )


class ItemMatchRequest(Request):
    """Item Match请求"""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        title="用户查询",
        description="原始用户查询"
    )
    collection: Optional[str] = Field(
        title="匹配的集合",
        description="有的实现可以不区分匹配集合，那么可以不指定这一项",
        default=None
    )


class ItemMatchResponse(Response):
    """Item Match响应"""

    model_config = ConfigDict(extra="allow")

    matches: List[str] = Field(
        title="匹配结果列表",
        description="如果没有匹配到则为空列表",
        default_factory=list
    )
