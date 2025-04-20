# hello-mcp-python

* **MCP server**:
  - handles connection management
  - handles protocol compliance
  - message routing
* **Resources**: 
  - similar to GET endpoint
  - expose data to LLMs
* **Tools**: 
  - perform compuation
  - have side effects
* **Prompts**: 
  - reusable templates
  - help LLMs interact with your server effectively
* **Images**:
  - automatically handles image data
* **Context**:
  - gives your tools and resources access to MCP capabilities


## install `uv` if not installed
```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ uv add "mcp[cli]"
$ # append the following to ~/.zshrc
export PATH=$HOME/.local/bin:.venv/bin:$PATH 
```

## Run hello mcp
```
$ mcp dev server.py # to test with MCP inspector
```

## Run leave-manager 
```
$ mcp dev leave-manager.py # test with MCP inspector
$ mcp install main.py
```


