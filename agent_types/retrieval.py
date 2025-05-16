#!/usr/bin/env python3

__author__ = "xi"

from typing import Dict, List, Optional, Union

from pydantic import ConfigDict, Field

from agent_types.common import CSRArray, NDArray, Request, Response


class DenseEmbeddingRequest(Request):
    """稠密向量表征请求"""

    model_config = ConfigDict(extra="allow")

    text: Union[str, List[str]] = Field(
        title="文本",
        description="需要表征的文本"
    )
    normalize: bool = Field(
        title="是否规一化",
        description="是否对结果向量进行规一化",
        default=True
    )


class DenseEmbeddingResponse(Response):
    """稠密向量表征响应"""

    model_config = ConfigDict(extra="allow")

    embedding: NDArray = Field(
        title="表征向量",
        description="稠密表征向量（若表征多条文本，则返回矩阵）"
    )


class SparseEmbeddingRequest(Request):
    """稀疏向量表征请求"""

    model_config = ConfigDict(extra="allow")

    text: Union[str, List[str]] = Field(
        title="文本",
        description="需要表征的文本"
    )


class SparseEmbeddingResponse(Response):
    """稀疏向量表征响应"""

    model_config = ConfigDict(extra="allow")

    embedding: CSRArray = Field(
        title="表征向量",
        description="稀疏表征向量（若表征多条文本，则返回矩阵）"
    )


class RetrievalRequest(Request):
    """知识库检索请求"""

    model_config = ConfigDict(extra="allow")

    collections: List[str] = Field(
        title="检索集合名称",
        description="检索集合名称，多个取值表示同时从多个集合中检索"
    )
    query: str = Field(
        title="用户查询",
        description="检索所需的用户查询"
    )
    top_k: int = Field(
        title="检索返回Top-k",
        description="检索返回Top-k",
        default=6,
        ge=1,
        le=50
    )
    expr: Dict[str, str] = Field(
        title="元数据表达式",
        description="元数据表达式，即筛选条件",
        default_factory=dict
    )
    index_fields: List[str] = Field(
        title="索引字段名称",
        description="索引字段的名称，默认为vector，可以指定多个表示混合检索",
        default=("vector",)
    )
    output_fields: Optional[List[str]] = Field(
        title="返回字段",
        description="检索结果中需包含的字段列表, 默认返回除向量字段以外的所有字段",
        default=None
    )
    use_rerank: bool = Field(
        title="是否使用Rerank",
        description="是否使用Rerank",
        default=False
    )
    pre_top_k: Optional[int] = Field(
        title="Rerank候选集大小",
        description="Rerank候选集大小",
        default=None,
        ge=1,
        le=50
    )


class RetrievalResponse(Response):
    """知识库检索响应"""

    model_config = ConfigDict(extra="allow")

    distance: List[float] = Field(
        title="检索距离",
        description="检索距离",
        default_factory=list
    )
    scores: List[float] = Field(
        title="重排分数",
        description="重排分数",
        default_factory=list
    )
    items: List[Dict] = Field(
        title="检索对象列表",
        description="检索对象列表",
        default_factory=list
    )
