# ai

## Errors / Solutions
* Error: spawn uv ENOENT (With Claude)
  Solution: Change command from `uv` to `/absolute/path/to/uv`
  ```
  {
    "mcpServers": {
      "LeaveManager": {
        "command": "/Users/allenkim/.local/bin/uv",
        "args": [...]
      }
    }
  }
  ```