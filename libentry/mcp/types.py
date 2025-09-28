#!/usr/bin/env python3

__author__ = "xi"

import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

JSONObject = Dict[str, Any]
JSONList = List[Any]
JSONType = Union[JSONObject, JSONList, str, int, float, bool, None]


class MIME(Enum):
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


class SubroutineError(BaseModel):
    error: str
    message: str
    traceback: str

    @classmethod
    def from_exception(cls, e):
        err_cls = e.__class__
        err_name = err_cls.__name__
        module = err_cls.__module__
        if module != "builtins":
            err_name = f"{module}.{err_name}"
        return cls(
            error=err_name,
            message=str(e),
            traceback=traceback.format_exc()
        )


class SubroutineResponse(BaseModel):
    result: Optional[Any] = None
    error: Optional[SubroutineError] = None


class HTTPOptions(BaseModel):
    method: Literal["GET", "POST"] = "POST"
    headers: Optional[Dict[str, str]] = None
    stream: Optional[bool] = None
    timeout: int = 15
    num_trials: int = 5
    interval: float = 1
    retry_factor: float = 0.5
    on_error: Optional[Callable[[Exception], None]] = None


class HTTPRequest(BaseModel):
    """HTTP request for a single trial"""

    path: str
    json_obj: Optional[Dict[str, Any]] = None
    options: HTTPOptions = Field(default_factory=HTTPOptions)


class SSE(BaseModel):
    """Server Send Event"""

    event: str
    data: Optional[Any] = None


class HTTPResponse(BaseModel):
    """HTTP response"""

    status_code: int
    headers: Dict[str, str]
    stream: bool
    content: Any = None


class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: Union[str, int]
    method: str
    params: Union[Dict[str, Any], None] = None

    model_config = ConfigDict(extra="allow")


class JSONRPCNotification(BaseModel):
    jsonrpc: Literal["2.0"]
    method: str
    params: Union[Dict[str, Any], None] = None

    model_config = ConfigDict(extra="allow")


class JSONRPCError(BaseModel):
    code: int = 0
    message: str
    data: Optional[Any] = None

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
    jsonrpc: Literal["2.0"]
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None

    model_config = ConfigDict(extra="allow")


class PaginatedRequest(BaseModel):
    cursor: Optional[str] = None
    """
    An opaque token representing the current pagination position.
    If provided, the server should return results starting after this cursor.
    """


class PaginatedResult(BaseModel):
    nextCursor: Optional[str] = None
    """
    An opaque token representing the pagination position after the last returned result.
    If present, there may be more results available.
    """


class Implementation(BaseModel):
    """Describes the name and version of an MCP implementation."""

    name: str
    version: str
    model_config = ConfigDict(extra="allow")


class RootsCapability(BaseModel):
    """Capability for root operations."""

    listChanged: Optional[bool] = None
    """Whether the client supports notifications for changes to the roots list."""
    model_config = ConfigDict(extra="allow")


class SamplingCapability(BaseModel):
    """Capability for logging operations."""

    model_config = ConfigDict(extra="allow")


class ClientCapabilities(BaseModel):
    """Capabilities a client may support."""

    experimental: Optional[Dict[str, Dict[str, Any]]] = None
    """Experimental, non-standard capabilities that the client supports."""
    sampling: Optional[SamplingCapability] = None
    """Present if the client supports sampling from an LLM."""
    roots: Optional[RootsCapability] = None
    """Present if the client supports listing roots."""
    model_config = ConfigDict(extra="allow")


class PromptsCapability(BaseModel):
    """Capability for prompts operations."""

    listChanged: Optional[bool] = None
    """Whether this server supports notifications for changes to the prompt list."""
    model_config = ConfigDict(extra="allow")


class ResourcesCapability(BaseModel):
    """Capability for resources operations."""

    subscribe: Optional[bool] = None
    """Whether this server supports subscribing to resource updates."""
    listChanged: Optional[bool] = None
    """Whether this server supports notifications for changes to the resource list."""
    model_config = ConfigDict(extra="allow")


class ToolsCapability(BaseModel):
    """Capability for tools operations."""

    listChanged: Optional[bool] = None
    """Whether this server supports notifications for changes to the tool list."""
    model_config = ConfigDict(extra="allow")


class LoggingCapability(BaseModel):
    """Capability for logging operations."""

    model_config = ConfigDict(extra="allow")


class ServerCapabilities(BaseModel):
    """Capabilities that a server may support."""

    experimental: Optional[Dict[str, Dict[str, Any]]] = None
    """Experimental, non-standard capabilities that the server supports."""
    logging: Optional[LoggingCapability] = None
    """Present if the server supports sending log messages to the client."""
    prompts: Optional[PromptsCapability] = None
    """Present if the server offers any prompt templates."""
    resources: Optional[ResourcesCapability] = None
    """Present if the server offers any resources to read."""
    tools: Optional[ToolsCapability] = None
    """Present if the server offers any tools to call."""
    model_config = ConfigDict(extra="allow")


class InitializeRequestParams(BaseModel):
    """Parameters for the initialize request."""

    protocolVersion: Union[str, int]
    """The latest version of the Model Context Protocol that the client supports."""
    capabilities: ClientCapabilities
    clientInfo: Implementation
    model_config = ConfigDict(extra="allow")


class InitializeResult(BaseModel):
    """After receiving an initialize request from the client, the server sends this."""

    protocolVersion: Union[str, int]
    """The version of the Model Context Protocol that the server wants to use."""
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: Optional[str] = None
    """Instructions describing how to use the server and its features."""


class InitializedNotification(JSONRPCNotification):
    """
    This notification is sent from the client to the server after initialization has
    finished.
    """

    method: Literal["notifications/initialized"]


class ToolAnnotations(BaseModel):
    """
    Additional properties describing a Tool to clients.

    NOTE: all properties in ToolAnnotations are **hints**.
    They are not guaranteed to provide a faithful description of
    tool behavior (including descriptive properties like `title`).

    Clients should never make tool use decisions based on ToolAnnotations
    received from untrusted servers.
    """

    title: Optional[str] = None
    """A human-readable title for the tool."""

    readOnlyHint: Optional[bool] = None
    """
    If true, the tool does not modify its environment.
    Default: false
    """

    destructiveHint: Optional[bool] = None
    """
    If true, the tool may perform destructive updates to its environment.
    If false, the tool performs only additive updates.
    (This property is meaningful only when `readOnlyHint == false`)
    Default: true
    """

    idempotentHint: Optional[bool] = None
    """
    If true, calling the tool repeatedly with the same arguments 
    will have no additional effect on the its environment.
    (This property is meaningful only when `readOnlyHint == false`)
    Default: false
    """

    openWorldHint: Optional[bool] = None
    """
    If true, this tool may interact with an "open world" of external
    entities. If false, the tool's domain of interaction is closed.
    For example, the world of a web search tool is open, whereas that
    of a memory tool is not.
    Default: true
    """
    model_config = ConfigDict(extra="allow")


class ToolProperty(BaseModel):
    type: Optional[str] = None
    anyOf: Optional[List["ToolProperty"]] = None
    items: Optional["ToolProperty"] = None
    properties: Optional[Dict[str, "ToolProperty"]] = None
    additionalProperties: Optional[Union[bool, "ToolProperty"]] = None
    # title: Optional[str] = None
    description: Optional[str] = None
    required: Optional[List[str]] = None


class ToolSchema(BaseModel):
    type: str = "object"
    properties: Optional[Dict[str, "ToolProperty"]] = None
    # title: Optional[str] = None
    description: Optional[str] = None
    required: Optional[List[str]] = None


class Tool(BaseModel):
    """Definition for a tool the client can call."""

    name: str
    """The name of the tool."""
    description: Optional[str] = None
    """A human-readable description of the tool."""
    inputSchema: Optional[ToolSchema] = None
    """A JSON Schema object defining the expected parameters for the tool."""
    annotations: Optional[ToolAnnotations] = None
    """Optional additional tool information."""
    model_config = ConfigDict(extra="allow")


class ListToolsResult(PaginatedResult):
    """The server's response to a tools/list request from the client."""

    tools: List[Tool]


class TextContent(BaseModel):
    """Text content for a message."""

    type: Literal["text"] = "text"
    text: str
    """The text content of the message."""

    model_config = ConfigDict(extra="allow")


class ImageContent(BaseModel):
    """Image content for a message."""

    type: Literal["image"] = "image"
    data: str
    """The base64-encoded image data."""
    mimeType: str
    """
    The MIME type of the image. Different providers may support different
    image types.
    """

    model_config = ConfigDict(extra="allow")


class AudioContent(BaseModel):
    """Audio content for a message."""

    type: Literal["audio"]
    data: str
    """The base64-encoded audio data."""
    mimeType: str
    """
    The MIME type of the audio. Different providers may support different
    audio types.
    """

    model_config = ConfigDict(extra="allow")


class CallToolRequestParams(BaseModel):
    """Parameters for calling a tool."""

    name: str
    arguments: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class CallToolResult(BaseModel):
    """The server's response to a tool call."""

    content: List[Union[TextContent, ImageContent, AudioContent]] = []
    structuredContent: Optional[Dict[str, Any]] = None
    isError: bool = False

    @classmethod
    def from_exception(cls, e: Exception):
        error = SubroutineError.from_exception(e)
        return cls(
            content=[TextContent(text=error.message)],
            structuredContent=error.model_dump(exclude_none=True),
            isError=True
        )

    @classmethod
    def from_subroutine_response(cls, response: SubroutineResponse):
        if response.error is not None:
            error = response.error
            return cls(
                content=[TextContent(text=error.message)],
                structuredContent=error.model_dump(exclude_none=True),
                isError=True
            )
        else:
            result = response.result
            if isinstance(result, Dict):
                return cls(
                    content=[],
                    structuredContent=result,
                    isError=False
                )
            elif isinstance(result, BaseModel):
                return cls(
                    content=[],
                    structuredContent=result.model_dump(exclude_none=False),
                    isError=False
                )
            elif isinstance(result, List):
                return cls(
                    content=result,
                    structuredContent=None,
                    isError=False
                )
            elif isinstance(result, (TextContent, ImageContent, AudioContent)):
                return cls(
                    content=[result],
                    structuredContent=None,
                    isError=False
                )
            elif isinstance(result, str):
                return cls(
                    content=[TextContent(text=result)],
                    structuredContent=None,
                    isError=False
                )
            else:
                return cls(
                    content=[TextContent(text=str(result))],
                    structuredContent=None,
                    isError=False
                )


class Resource(BaseModel):
    """A known resource that the server is capable of reading."""

    uri: str
    """The URI of this resource."""
    name: str
    """A human-readable name for this resource."""
    description: Optional[str] = None
    """A description of what this resource represents."""
    mimeType: Optional[str] = None
    """The MIME type of this resource, if known."""
    size: Optional[int] = None
    """
    The size of the raw resource content, in bytes (i.e., before base64 encoding
    or any tokenization), if known.

    This can be used by Hosts to display file sizes and estimate context window usage.
    """

    model_config = ConfigDict(extra="allow")


class ListResourcesResult(PaginatedResult):
    """The server's response to a resources/list request from the client."""

    resources: List[Resource]


class ReadResourceRequestParams(BaseModel):
    """Parameters for reading a resource."""

    uri: str
    """
    The URI of the resource to read. The URI can use any protocol; it is up to the
    server how to interpret it.
    """

    model_config = ConfigDict(extra="allow")


class ResourceContents(BaseModel):
    """The contents of a specific resource or sub-resource."""

    uri: str
    """The URI of this resource."""
    mimeType: Optional[str] = None
    """The MIME type of this resource, if known."""

    model_config = ConfigDict(extra="allow")


class TextResourceContents(ResourceContents):
    """Text contents of a resource."""

    text: str
    """
    The text of the item. This must only be set if the item can actually be represented
    as text (not binary data).
    """


class BlobResourceContents(ResourceContents):
    """Binary contents of a resource."""

    blob: str
    """A base64-encoded string representing the binary data of the item."""


class ReadResourceResult(BaseModel):
    """The server's response to a resources/read request from the client."""

    contents: List[Union[TextResourceContents, BlobResourceContents]]
