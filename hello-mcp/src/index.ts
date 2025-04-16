import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// Initialize server with resource capabilities
const server = new Server(
  {
    name: "hello-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      resources: {}, // Enable resources
    },
  }
);
// List available resources when clients request them
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "hello://world",
        name: "Hello World Message",
        description: "A simple greeting message",
        mimeType: "text/plain",
      },
    ],
  };
});
// Return resource content when clients request it
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  if (request.params.uri === "hello://world") {
    return {
      contents: [
        {
          uri: "hello://world",
          text: "Hello, World! This is my first MCP resource.",
        },
      ],
    };
  }

  const greetingExp = /^greetings:\/\/(.+)$/;
  const greetingMatch = request.params.uri.match(greetingExp);
  if (greetingMatch) {
    const name = decodeURIComponent(greetingMatch[1]);
    return {
        contents: [
        {
            uri: request.params.uri,
            text: `Hello, ${name}! Welcome to MCP.`,
        },
      ],
    };
  }

  throw new Error("Resource not found");
});

// Resource Templates
server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => {
  return {
    resourceTemplates: [
      {
        uriTemplate: 'greetings://{name}',
        name: 'Personal Greeting',
        description: 'A personalized greeting message',
        mimeType: 'text/plain',
      },
    ],
  };
});

// Start server using stdio transport
const transport = new StdioServerTransport();
await server.connect(transport);
console.info('{"jsonrpc": "2.0", "method": "log", "params": { "message": "Server running..." }}');