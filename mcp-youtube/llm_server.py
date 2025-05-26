import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
    handlers=[logging.StreamHandler()]
)

# Create a logger instance
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MCP_TOOL_URL = "http://localhost:8000/tools/get_youtube_transcript"
MCP_SYSTEM_PROMPT = """
You are an assistant that strictly follows tool-calling instructions.

If the user's query contains a YouTube URL (e.g., https://youtu.be/ or https://youtube.com/),
you MUST respond with EXACTLY this format and NOTHING else:

call_tool: get_youtube_transcript <video_url>

Replace <video_url> with the full YouTube URL provided in the user's query (e.g., https://youtube.com/watch?v=dQw4w9WgXcQ).
Do NOT use placeholders like <video_url>.
The response must be the exact tool call line with the full YouTube URL and nothing more.

For all other queries without using a tool, respond with a concise text answer.
Do NOT include any reasoning, explanations, internal thought processes, or tags such as <think> or </think>.
Provide only the final answer in plain text, avoiding any unnecessary details or formatting.

Available tools:
- get_youtube_transcript <video_url>
"""

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: PromptRequest):
    user_prompt = req.prompt
    full_prompt = f"{MCP_SYSTEM_PROMPT}\\n\\nUser: {user_prompt}"
    logger.info(f"user_prompt: {user_prompt}")
    logger.debug(f"full_prompt: {full_prompt}")

    # Send prompt to Ollama
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(OLLAMA_URL, json={
                "model": "DeepSeek-R1-7B", "prompt": full_prompt, "stream": False
            })
            response.raise_for_status()  # Raise exception for HTTP errors

            data = response.json()
            response_text = data.get("response", "")
            logger.info(f"Response from Ollama: {response_text}")
            response_text = response_text.split('</think>')[1].strip() # Remove the <think> tag if present

            if "call_tool:" in response_text:
                # Extract the line starting with "call_tool:"
                tool_call_line = next((line for line in response_text.splitlines() if line.startswith("call_tool:")), None)
                if tool_call_line:
                    parts = tool_call_line.strip().split()
                    logger.debug(f"call_tool requested: {parts}")
                else:
                    logger.error("No valid tool call line found.")
                    return {"response": "Invalid tool call format."}
            else:
                return {"response": response_text}

            if parts[1] == "get_youtube_transcript":
                video_url = parts[2]
                transcript_res = await client.post(MCP_TOOL_URL, json={"video_url": video_url})
                result = transcript_res.json().get("result", [])
                if result and "text" in result[0]:
                    transcript = result[0]["text"]
                else:
                    logger.error("Failed to retrieve transcript.")
                    return {"response": "Transcript retrieval failed."}

                return {"response": transcript}
            else:
                logger.error("Invalid tool call format.")
                return {"response": "Invalid tool call format."}

    except httpx.RequestError as e:
        logger.error(f"Request error: {str(e)}")
        return {"response": f"Failed to connect to external service: {str(e)}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return {"response": f"External service error: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"response": "An unexpected error occurred."}

if __name__ == "__main__":
    logger.info("Starting FastAPI LLM server with Uvicorn")
    uvicorn.run("llm_server:app", host="0.0.0.0", port=3000, reload=True)
