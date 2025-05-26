# fastapi_server.py
import re
import logging
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
import uvicorn
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Define mcp
try:
    logging.info("Initializing FastMCP server")
    mcp = FastMCP(
        "YouTube Transcript Extractor",
        dependencies=["youtube-transcript-api>=0.6.2"]
    )
except Exception as e:
    logging.error(f"Failed to initialize FastMCP: {e}")
    raise

def _extract_video_id(url: str) -> str | None:
    """Extracts the YouTube video ID from various URL formats."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    match = re.search(r"youtu\.be\/([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)
    match = re.search(r"\/embed\/([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)
    return None

@mcp.tool()
def get_youtube_transcript(video_url: str) -> str:
    """
    Fetches the transcript for a given YouTube video URL.
    Args:
        video_url: The URL of the YouTube video.
    Returns:
        A string containing the full transcript, or an error message.
    """
    video_id = _extract_video_id(video_url)
    if not video_id:
        return "Error: Could not extract Video ID from the provided URL."
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([item["text"] for item in transcript_list])
        return transcript_text
    except TranscriptsDisabled:
        return f"Error: Transcripts are disabled for video ID: {video_id}"
    except NoTranscriptFound:
        return f"Error: No transcript found for video ID: {video_id}."
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"

# Run the MCP server
app = FastAPI(title="YouTube Transcript Extractor API") # Initialize FastAPI app

@app.post("/tools/get_youtube_transcript")  # FastAPI endpoint to call the MCP tool
async def call_youtube_transcript(data: dict):
    video_url = data.get("video_url")
    if not video_url:
        return {"result": "Error: No video_url provided"}
    
    try:
        result = await mcp.call_tool("get_youtube_transcript", {"video_url": video_url})
        return {"result": result}
    except Exception as e:
        return {"result": f"Error: Failed to call tool: {str(e)}"}

if __name__ == "__main__":  # Start the FastAPI server with Uvicorn
    logging.info("Starting FastAPI server with Uvicorn")
    uvicorn.run(app, host="127.0.0.1", port=8000)
    mcp.run