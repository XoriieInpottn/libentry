#!/usr/bin/env python3

__author__ = "xi"

import traceback
import uuid
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ContentType(Enum):
    plain = "text/plain"
    form = "application/x-www-form-urlencoded"
    html = "text/html"
    # object type
    xml = "application/xml"
    json = "application/json"
    # stream type
    octet_stream = "application/octet-stream"
    json_stream = "application/json-stream"
    sse = "text/event-stream"


class _JSONRequest(BaseModel):
    method: Literal["GET", "POST"]
    path: str
    json_obj: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    stream: Optional[bool] = None
    timeout: int = 15


class JSONRequest(_JSONRequest):
    num_trials: int = 5
    interval: float = 1
    retry_factor: float = 0.5
    on_error: Optional[Callable[[Exception], None]] = None


class SSE(BaseModel):
    event: str = Field()
    data: Optional[Any] = Field(default=None)


class JSONResponse(BaseModel):
    status_code: int
    headers: Dict[str, str]
    stream: bool
    content: Any = None


class JSONRPCRequest(BaseModel):
    jsonrpc: str = Field(default="2.0")
    id: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str = Field()
    params: Optional[Dict[str, Any]] = Field(default=None)


class JSONRPCError(BaseModel):
    code: int = Field(default=0)
    message: str = Field()
    data: Optional[Any] = Field(default=None)

    @classmethod
    def from_exception(cls, e):
        err_cls = e.__class__
        err_name = err_cls.__name__
        module = err_cls.__module__
        if module != "builtins":
            err_name = f"{module}.{err_name}"
        return cls(
            code=0,
            message=str(e),
            data={
                "error": err_name,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )


class JSONRPCResponse(BaseModel):
    jsonrpc: str = Field(default="2.0")
    id: Union[str, int] = Field()
    result: Optional[Any] = Field(default=None)
    error: Optional[JSONRPCError] = Field(default=None)


class JSONRPCNotification(BaseModel):
    jsonrpc: str = Field(default="2.0")
    method: str = Field()
    params: Optional[Dict[str, Any]] = Field(default=None)


class ServiceError(RuntimeError):

    def __init__(self, message: str, cause: Optional[str] = None, _traceback: Optional[str] = None):
        self.message = message
        self.cause = cause
        self.traceback = _traceback

    def __str__(self):
        lines = []
        if self.message:
            lines += [self.message, "\n\n"]
        if self.cause:
            lines += ["This is caused by server side error ", self.cause, ".\n"]
        if self.traceback:
            lines += ["Below is the stacktrace:\n", self.traceback.rstrip()]
        return "".join(lines)

    @staticmethod
    def from_rpc_error(error: JSONRPCError):
        cause = None
        traceback_ = None
        if isinstance(error.data, Dict):
            cause = error.data.get("error")
            traceback_ = error.data.get("traceback")
        return ServiceError(error.message, cause, traceback_)


class Implementation(BaseModel):
    """Describes the name and version of an MCP implementation."""

    name: str
    version: str
    model_config = ConfigDict(extra="allow")


class RootsCapability(BaseModel):
    """Capability for root operations."""

    listChanged: bool | None = None
    """Whether the client supports notifications for changes to the roots list."""
    model_config = ConfigDict(extra="allow")


class SamplingCapability(BaseModel):
    """Capability for logging operations."""

    model_config = ConfigDict(extra="allow")


class ClientCapabilities(BaseModel):
    """Capabilities a client may support."""

    experimental: dict[str, dict[str, Any]] | None = None
    """Experimental, non-standard capabilities that the client supports."""
    sampling: SamplingCapability | None = None
    """Present if the client supports sampling from an LLM."""
    roots: RootsCapability | None = None
    """Present if the client supports listing roots."""
    model_config = ConfigDict(extra="allow")


class PromptsCapability(BaseModel):
    """Capability for prompts operations."""

    listChanged: bool | None = None
    """Whether this server supports notifications for changes to the prompt list."""
    model_config = ConfigDict(extra="allow")


class ResourcesCapability(BaseModel):
    """Capability for resources operations."""

    subscribe: bool | None = None
    """Whether this server supports subscribing to resource updates."""
    listChanged: bool | None = None
    """Whether this server supports notifications for changes to the resource list."""
    model_config = ConfigDict(extra="allow")


class ToolsCapability(BaseModel):
    """Capability for tools operations."""

    listChanged: bool | None = None
    """Whether this server supports notifications for changes to the tool list."""
    model_config = ConfigDict(extra="allow")


class LoggingCapability(BaseModel):
    """Capability for logging operations."""

    model_config = ConfigDict(extra="allow")


class ServerCapabilities(BaseModel):
    """Capabilities that a server may support."""

    experimental: dict[str, dict[str, Any]] | None = None
    """Experimental, non-standard capabilities that the server supports."""
    logging: LoggingCapability | None = None
    """Present if the server supports sending log messages to the client."""
    prompts: PromptsCapability | None = None
    """Present if the server offers any prompt templates."""
    resources: ResourcesCapability | None = None
    """Present if the server offers any resources to read."""
    tools: ToolsCapability | None = None
    """Present if the server offers any tools to call."""
    model_config = ConfigDict(extra="allow")


class InitializeRequestParams(BaseModel):
    """Parameters for the initialize request."""

    protocolVersion: str | int
    """The latest version of the Model Context Protocol that the client supports."""
    capabilities: ClientCapabilities
    clientInfo: Implementation
    model_config = ConfigDict(extra="allow")


class InitializeResult(BaseModel):
    """After receiving an initialize request from the client, the server sends this."""

    protocolVersion: str | int
    """The version of the Model Context Protocol that the server wants to use."""
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: str | None = None
    """Instructions describing how to use the server and its features."""


class InitializedNotification(JSONRPCNotification):
    """
    This notification is sent from the client to the server after initialization has
    finished.
    """

    method: Literal["notifications/initialized"]


class TextContent(BaseModel):
    """Text content for a message."""

    type: Literal["text"] = "text"
    text: str
    """The text content of the message."""
    model_config = ConfigDict(extra="allow")


class CallToolRequestParams(BaseModel):
    """Parameters for calling a tool."""

    name: str
    arguments: dict[str, Any] | None = None
    model_config = ConfigDict(extra="allow")


class CallToolResult(BaseModel):
    """The server's response to a tool call."""

    content: list[TextContent]
    isError: bool = False
