# server.py
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from PIL import Image as PILImage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an MCP server
mcp = FastMCP("Hello MCP")

@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"

@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?")
    ]

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    img = PILImage.open(image_path)
    logger.info(f"Original size: {img.size}")
    img.thumbnail((100, 100))
    logger.info(f"Thumbnail size: {img.size}")
    img_data = img.tobytes()
    logger.info(f"Image data length: {len(img_data)}")
    return Image(data=img_data, format="png")