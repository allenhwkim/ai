import re
from mcp.server.fastmcp import FastMCP
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# Create an MCP server instance
mcp = FastMCP(
    "YouTube Transcript Extractor", 
    dependencies=["youtube-transcript-api>=0.6.2"]
)

# Standard watch URL: https://www.youtube.com/watch?v=VIDEO_ID
def _extract_video_id(url: str) -> str | None:
    """Extracts the YouTube video ID from various URL formats."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)

    # Shortened URL: https://youtu.be/VIDEO_ID
    match = re.search(r"youtu\.be\/([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)

    # Embed URL: https://www.youtube.com/embed/VIDEO_ID
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
        A string containing the full transcript, with each segment on a new line,
        or an error message if the transcript cannot be fetched.
    """
    video_id = _extract_video_id(video_url)

    if not video_id:
        return "Error: Could not extract Video ID from the provided URL."

    try:
        # Fetch the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Format the transcript into a single string
        transcript_text = "\n".join([item["text"] for item in transcript_list])
        return transcript_text

    except TranscriptsDisabled:
        return f"Error: Transcripts are disabled for video ID: {video_id}"
    except NoTranscriptFound:
        return f"Error: No transcript found for video ID: {video_id}. The video might not have subtitles or they are not in a supported language."
    except Exception as e:
        # Catch any other potential exceptions from the API
        return f"Error: An unexpected error occurred while fetching the transcript for video ID {video_id}: {str(e)}"


if __name__ == "__main__":
    mcp.run()