# hello-mcp

When you run this and connect`$ npx @modelcontextprotocol/inspector build/index.js`

**Error in /sse route: Error: spawn build/index.js EACCES**
Solution: $ chmod 755 build/index.js # when you see permission error

**Error in /sse route: Error: spawn Unknown system error -8**
Solution: Add `#!/usr/bin/env node` on the top of `build/index.js`
