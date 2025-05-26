Install packages
```
$ uv sync
$ source .venv/bin/activateDefine mcp
```

Run mcp server
```
$ uv run mcp_server.py
$ curl -X POST http://127.0.0.1:8000/tools/get_youtube_transcript --json '{"video_url": "https://www.youtube.com/watch?v=_blFagKJhks"}'
```

Run LLM server
```
$ uv run llm_server.py
$ curl -X POST http://localhost:3000/chat --json '{"prompt": "How are you?"}'
$ curl -X POST http://localhost:3000/chat --json '{"prompt": "Summarize https://youtube.com/watch?v=dQw4w9WgXcQ"}'
```

Call llm server from index.html
```
$ http-server .
```