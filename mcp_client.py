import asyncio
from asyncio import sleep

import rich
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


async def main():
    try:
        async with sse_client(url="http://localhost:8000/sse") as (read, write):
            # 创建客户端会话
            async with ClientSession(read, write) as session:
                # 初始化会话
                await session.initialize()

                # List available tools
                tools = await session.list_tools()
                rich.print(tools)

                resources = await session.list_resources()
                rich.print(resources)

                # Call the fetch tool
                result = await session.call_tool("add_stream", {"a": 1, "b": 2})
                rich.print(result)
    except Exception as e:
        rich.print(f"[bold red]Error:[/bold red] {str(e)}")


asyncio.run(main())
