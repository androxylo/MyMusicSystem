"""
MCP server — exposes all agent_tools as Claude-callable tools.

Run via:
    uv run --python 3.11 mcp_server.py

Or through Claude Code (configured in .claude/settings.local.json).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
import agent_tools.tools as t

mcp = FastMCP("MyMusicSystem")

mcp.tool()(t.start_session)
mcp.tool()(t.rate_track)
mcp.tool()(t.complete_session)
mcp.tool()(t.get_stats)
mcp.tool()(t.list_engines)
mcp.tool()(t.get_track_info)
mcp.tool()(t.get_playlists)
mcp.tool()(t.set_curated_threshold)
mcp.tool()(t.reconcile_playlists)

if __name__ == "__main__":
    mcp.run()
