import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  GetPromptRequestSchema,
  ListPromptsRequestSchema,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListToolsRequestSchema,
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
      prompts: {},
      resources: {}, // Enable resources
      tools: {},
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

// Prompts
server.setRequestHandler(ListPromptsRequestSchema, () => {
  return {
    prompts: [
      {
        name: "create-greeting",
        description: "Generate a customized greeting message",
        arguments: [
          {
            name: "name",
            description: "Name of the person to greet",
            required: true,
          },
          {
            name: "style",
            description: "The style of greeting, such a formal, excited, or casual. If not specified casual will be used"
          }
        ],
      }
    ]
  }
});

server.setRequestHandler(GetPromptRequestSchema, (request) => {
  const { name, arguments: args } = request.params;

  const promptHandlers = {
    "create-greeting": ({ name, style = "casual" }: { name: string, style?: string }) => {
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Please generate a greeting in ${style} style to ${name}.`,
            },
          },
        ],
      };
    },
  };

  const promptHandler = promptHandlers[name as keyof typeof promptHandlers];
  if (promptHandler) return promptHandler(args as { name: string, style?: string });
  throw new Error("Prompt not found");
});

// tools 
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'create-message',
        description: 'Generate a custom message with various options',
        inputSchema: {
          type: 'object',
          properties: {
            messageType: {
              type: 'string',
              enum: ['greeting', 'farewell', 'thank-you'] ,
              description: 'Type of message to generate',
            },
            recipient: {
              type: 'string',
              description: 'Name of the person to address',
            },
            tone: {
              type: 'string',
              enum: ['formal', 'casual', 'playful'],
              description: 'Tone of the message',
            },
          },
          required: ['messageType', 'recipient'],
        },
      }
    ],
  }
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {

  const { name, arguments: params } = request.params ?? {};
  if (name === 'create-message') {
    const handler = (args: { messageType: string; recipient: string; tone?: string; }) => {
        if (!args.messageType) throw new Error("Must provide a message type.");
        if (!args.recipient) throw new Error("Must provide a recipient.");
        const { messageType, recipient } = args;
        const tone = args.tone || "casual";
        const messageFns = {
          greeting: {
            formal: (recipient: string) => `Dear ${recipient}, I hope this message finds you well`,
            playful: (recipient: string) => `Hey hey ${recipient}! 🎉 What's shakin'?`,
            casual: (recipient: string) => `Hi ${recipient}! How are you?`,
          },
          farewell: {
            formal: (recipient: string) => `Best regards, ${recipient}. Until we meet again.`,
            playful: (recipient: string) => `Catch you later, ${recipient}! 👋 Stay awesome!`,
            casual: (recipient: string) => `Goodbye ${recipient}, take care!`,
          },
          "thank-you": {
            formal: (recipient: string) => `Dear ${recipient}, I sincerely appreciate your assistance.`,
            playful: (recipient: string) => `You're the absolute best, ${recipient}! 🌟 Thanks a million!`,
            casual: (recipient: string) => `Thanks so much, ${recipient}! Really appreciate it!`,
          },
        } as any;
        const func = messageFns[messageType][tone];
      
        if (func) {
          return {
            content: [
              {
                type: "text",
                text: func(recipient),
              },
            ],
          };
        }

        throw new Error( `Invalid message type of tone`);
    };
    return handler(...[params] as Parameters<typeof handler>)
  } else {
    throw new Error('Tool not found');
  }
});

// Start server using stdio transport
const transport = new StdioServerTransport();
await server.connect(transport);
console.info('{"jsonrpc": "2.0", "method": "log", "params": { "message": "Server running..." }}');