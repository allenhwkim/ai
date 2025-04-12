# First local AI agent using DeepSeek

## Download .gguf file
```
$ pip install huggingface-hub

$ huggingface-cli download \
  bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF \
  DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
  --local-dir ./DeepSeek-R1-7B \
  --local-dir-use-symlinks False

$ ls DeepSeek-R1-7B 
DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
```

## Create Modelfile
```
$ cd DeepSeek-R1-7B
$ vi Modelfile
FROM DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf

TEMPLATE """{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
"""

SYSTEM """A chat between a curious user and an AI"""

PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop <s>
PARAMETER stop </s>
```

### Create llm model
```
$ ollama create DeepSeek-R1-7B -f ./Modelfile
gathering model components
copying file sha256:78272d8d32084548bd450394a560eb2d70de8232ab96a725769b1f9171235c1c 100%
parsing GGUF
using existing layer sha256:78272d8d32084548bd450394a560eb2d70de8232ab96a725769b1f9171235c1c
using existing layer sha256:6b70a2ad0d545ca50d11b293ba6f6355eff16363425c8b163289014cf19311fc
using existing layer sha256:149946c88b0b4d88b063daaabc09057acdd2adfdfcfcc9bbbe2128b74f41b0dd
using existing layer sha256:94c5e1b184983877acea5e687503e61f57a49c6fea81e280b1767ccf7ab0a0f0
writing manifest
success

$ ollama list
NAME                        ID              SIZE      MODIFIED
DeepSeek-R1-7B:latest       3885376a4646    4.7 GB    2 minutes ago
EEVE_Korean-10.8B:latest    6f4930ca1487    7.7 GB    About an hour ago
```

###  Run llm model
```
$ ollama run DeepSeek-R1-7B
>>> What's the capital of Korea?
It seems like you're asking me about the capital of Korea. The capital city is
Seoul.</<think>
Okay, so I need to figure out what the capital of Korea is. Hmm, I remember hearing
it's Seoul before, but let me think through this step by step.
```
