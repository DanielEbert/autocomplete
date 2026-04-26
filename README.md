# fast-llama

VS Code extension for local LLM code completion via llama-server.

## Features

- **Inline completions** - Ghost text suggestions as you type
- **Ring buffer context** - Captures nearby code around cursor for smarter completions
- **KV cache pre-warming** - Idle-time context embedding for faster responses
- **Prefix cache** - Rolling substring matching for instant cache hits
- **Speculative completion** -预测 next token when you hit Tab

## Requirements

- VS Code
- llama-server running
- A code model (qwen3.5-coder, etc.)

## Installation

```bash
npm install
```

## Configuration

Set `fastLlama.endpoint` in VS Code settings:

```json
{
  "fastLlama.endpoint": "http://localhost:28582"
}
```

Or use `.vscode/settings.json` in the project.

## Usage

1. Start llama-server with a code model:

   ```bash
   llama-server -m hf_hub_sloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q5_K_M -ngl 999 -c 24576 -fao n --port 28582 --host 127.0.0.1
   ```

2. Open a code file in VS Code

3. Completions appear automatically as ghost text

4. Press Tab to accept, Escape to dismiss

## How It Works

- **Context capture**: On cursor move >32 lines or file switch, captures surrounding code chunk
- **Infill API**: Sends prefix+suffix to Ollama's `/infill` endpoint
- **Debounce**: 20ms delay avoids mid-word requests
- **Abort**: Cancels pending requests when user keeps typing
