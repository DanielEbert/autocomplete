import * as vscode from 'vscode';

const MAX_PREFIX_LINES = 256;
const MAX_SUFFIX_LINES = 64;
const MAX_PREDICT_TOKENS = 48;
const CACHE_SIZE = 100;

// Ring Buffer Constants
const CHUNK_SIZE = 64;
const MOVE_THRESHOLD = 32;
const JACCARD_THRESHOLD = 0.9;
const MAX_BUFFER_CHUNKS = 3; // Keep context concise for speed


export function activate(context: vscode.ExtensionContext) {
    const provider = new FastCompletionProvider();
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider),
        vscode.window.onDidChangeTextEditorSelection(() => provider.triggerContextCheck()),
        vscode.window.onDidChangeActiveTextEditor(() => provider.triggerContextCheck());
    );

    const interval = setInterval(() => provider.periodicRingBufferUpdate(), 30000);
    context.subscriptions.push({ dispose: () => clearInterval(interval) });

    console.log("Llama Autocomplete activated.");
}

class FastCompletionProvider implements vscode.InlineCompletionItemProvider {
    private cache = new PrefixCache(CACHE_SIZE);
    private contextBuffer = new ContextBuffer(MAX_BUFFER_CHUNKS);
    private abortController: AbortController | null = null;
    private isRequestInProgress = false;

    private lastCapturedUri: string = "";
    private lastCapturedLine: number = -1;
    private settleTimer: NodeJS.Timeout | null = null;

    public async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[] | null> {
        // Bail out immediately if user is still typing
        if (token.isCancellationRequested) return null;

        // Debounce (20ms is the sweet spot: invisible to humans, drops 50% of mid-word HTTP requests)
        await new Promise(resolve => setTimeout(resolve, 20));
        if (token.isCancellationRequested) return null;

        // Detect word fragment before cursor (e.g., "se" from "se|")
        const linePrefix = document.lineAt(position.line).text.slice(0, position.character);
        const wordMatch = linePrefix.match(/[\w\d_]+$/);
        const wordPrefix = wordMatch ? wordMatch[0] : "";
        const completionRange = new vscode.Range(
            position.translate(0, -wordPrefix.length),
            position
        );

        // Prepend Ring Buffer context to the prefix
        const bufferContext = this.contextBuffer.getMergedContext();
        const prefix = bufferContext + this.getText(document, position, MAX_PREFIX_LINES, true);
        const suffix = this.getText(document, position, MAX_SUFFIX_LINES, false);

        // Check "Rolling Substring" Cache
        const cachedCompletion = this.cache.get(prefix);
        if (cachedCompletion) {
            // Speculate the future in the background and return instantly
            this.speculateFuture(prefix, suffix, cachedCompletion);
            return [new vscode.InlineCompletionItem(wordPrefix + cachedCompletion, completionRange)];
        }

        // Concurrency Lock: Wait for prior requests to clear, aborting if typing continues
        while (this.isRequestInProgress) {
            await new Promise(resolve => setTimeout(resolve, 10));
            if (token.isCancellationRequested) return null;
        }

        // Fetch from Server
        this.isRequestInProgress = true;
        try {
            const completion = await this.fetchInfill(prefix, suffix);
            if (!completion || token.isCancellationRequested) return null;

            this.cache.set(prefix, completion);
            this.speculateFuture(prefix, suffix, completion);

            return [new vscode.InlineCompletionItem(wordPrefix + completion, completionRange)];
        } finally {
            this.isRequestInProgress = false;
        }
    }

    /**
     * Entry point for all movement events.
     * Consolidates selection changes and editor switches.
     */
    public triggerContextCheck() {
        if (this.settleTimer) clearTimeout(this.settleTimer);

        // 300ms delay allows "Go to Definition" to open the new file,
        // scroll to the line, and set the selection.
        this.settleTimer = setTimeout(() => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const doc = editor.document;
            const pos = editor.selection.active;
            const currentUri = doc.uri.toString();

            // Logic: Is this a different file? OR is it >32 lines away in the same file?
            const isNewFile = currentUri !== this.lastCapturedUri;
            const isFarMove = Math.abs(pos.line - this.lastCapturedLine) > MOVE_THRESHOLD;

            if (isNewFile || isFarMove) {
                this.captureContext(editor, pos);
            }
        }, 300);
    }

    private captureContext(editor: vscode.TextEditor, pos: vscode.Position) {
        const doc = editor.document;
        this.lastCapturedUri = doc.uri.toString();
        this.lastCapturedLine = pos.line;

        const startLine = Math.max(0, pos.line - CHUNK_SIZE / 2);
        const endLine = Math.min(doc.lineCount - 1, startLine + CHUNK_SIZE);
        const fileName = doc.fileName.split(/[\\/]/).pop() || "file";

        const chunkText = doc.getText(new vscode.Range(startLine, 0, endLine, 0));

        // Quality check: don't buffer tiny fragments or empty space
        if (chunkText.trim().length > 20) {
            const chunkWithMeta = `// File: ${fileName}\n${chunkText}`;

            if (this.contextBuffer.addChunk(chunkWithMeta)) {
                console.log(`Context Captured: ${fileName} (Line ${startLine} - ${endLine})`);
                // Force a pre-warm so the server starts tokenizing this new file immediately
                this.periodicRingBufferUpdate();
            }
        }
    }

    // Pre-warms the KV Cache by sending context chunks during idle time
    public async periodicRingBufferUpdate() {
        if (this.isRequestInProgress) return;

        const context = this.contextBuffer.getMergedContext();
        if (!context) return;

        console.log("Pre-warming Llama KV cache...");
        // Send with n_predict: 0. This tokenizes and caches without generating.
        await this.fetchInfill(context, "", 0);
    }

    private async fetchInfill(prefix: string, suffix: string, n_predict = MAX_PREDICT_TOKENS): Promise<string | null> {
        // Cancel any pending HTTP request (actual or speculative)
        if (this.abortController) this.abortController.abort();
        this.abortController = new AbortController();

        const endpoint = vscode.workspace.getConfiguration('fastLlama').get('endpoint') + '/infill';

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input_prefix: prefix,
                    input_suffix: suffix,
                    n_predict: n_predict,
                    cache_prompt: true,
                    samplers: ["top_p", "infill"],
                    top_p: 0.95,
                    temperature: 0.1,
                    stop: ["<|file_separator|>", "\n\n"]  // stop suggestions on newline
                }),
                signal: this.abortController.signal
            });

            if (!response.ok) return null;

            const data = await response.json() as any;
            const content = data.content;
            if (!content) return content;
            // console.log(`prefix: <START>${prefix}<END>, suffix: <START>${suffix}<END>\n, content: <START>${content}<END>`)

            const match = content.match(/^.*?\n[ \t]*/s);
            return match ? match[0] : content;
        } catch (err: any) {
            if (err.name === 'AbortError') return null; // Expected behavior when typing fast
            console.error("Llama server error:", err);
            return null;
        }
    }

    private speculateFuture(prefix: string, suffix: string, currentCompletion: string) {
        // Assume user hits 'Tab'. Ask the server what comes NEXT.
        const futurePrefix = prefix + currentCompletion;

        // Fire and forget (Do not lock `isRequestInProgress`)
        this.fetchInfill(futurePrefix, suffix, MAX_PREDICT_TOKENS).then(futureCompletion => {
            if (futureCompletion) {
                this.cache.set(futurePrefix, futureCompletion);
            }
        });
    }

    private getText(doc: vscode.TextDocument, pos: vscode.Position, linesLimit: number, isPrefix: boolean): string {
        if (isPrefix) {
            const startLine = Math.max(0, pos.line - linesLimit);
            return doc.getText(new vscode.Range(startLine, 0, pos.line, pos.character));
        } else {
            const endLine = Math.min(doc.lineCount - 1, pos.line + linesLimit);
            const endChar = doc.lineAt(endLine).text.length;
            return doc.getText(new vscode.Range(pos.line, pos.character, endLine, endChar));
        }
    }
}

class ContextBuffer {
    private chunks: string[] = [];

    constructor(private maxChunks: number) { }

    public addChunk(newChunk: string): boolean {
        // 1. Extract significant lines for comparison (ignore the "File:" tag and empty lines)
        const getLines = (c: string) => c.split('\n')
            .map(l => l.trim())
            .filter(l => l.length > 10); // Ignore short lines/boilerplate

        const newLines = getLines(newChunk);
        if (newLines.length < 3) return false;

        for (let i = 0; i < this.chunks.length; i++) {
            const existingLines = getLines(this.chunks[i]);

            // Calculate intersection (how many lines are shared)
            const shared = newLines.filter(line => existingLines.includes(line));
            const overlapRatioNew = shared.length / newLines.length;
            const overlapRatioExisting = shared.length / existingLines.length;

            // SCENARIO A: The new chunk is a subset of an existing chunk
            // If > 80% of the new chunk is already known, ignore it.
            if (overlapRatioNew > 0.8) {
                return false;
            }

            // SCENARIO B: The new chunk is better/more complete than an existing chunk
            // If > 80% of an existing chunk is inside the new chunk, delete the old one.
            if (overlapRatioExisting > 0.8) {
                this.chunks.splice(i, 1);
                i--; // Adjust index after removal
                continue;
            }
        }

        // Add the new unique chunk
        this.chunks.push(newChunk);

        // Keep it within limits
        if (this.chunks.length > this.maxChunks) {
            this.chunks.shift();
        }
        return true;
    }

    public getMergedContext(): string {
        if (this.chunks.length === 0) return "";
        // Join with clear boundaries for the model
        return this.chunks.join("\n\n") + "\n\n";
    }
}

class PrefixCache {
    private cache = new Map<string, string>();

    constructor(private maxSize: number) { }

    public get(currentPrefix: string): string | undefined {
        // Exact match
        if (this.cache.has(currentPrefix)) return this.cache.get(currentPrefix);

        // "Rolling Substring" Match
        // If I typed "def" and cache has "de" -> "f sum():", I can slice it to " sum():"
        for (const [cachedPrefix, completion] of this.cache.entries()) {
            if (currentPrefix.startsWith(cachedPrefix)) {
                const newlyTypedChars = currentPrefix.slice(cachedPrefix.length);
                if (completion.startsWith(newlyTypedChars)) {
                    return completion.slice(newlyTypedChars.length);
                }
            }
        }
        return undefined;
    }

    public set(prefix: string, completion: string) {
        if (!completion.trim()) return; // Don't cache empty garbage

        this.cache.set(prefix, completion);

        // Basic LRU: Map remembers insertion order. Delete oldest.
        if (this.cache.size > this.maxSize) {
            const firstKey = this.cache.keys().next().value;
            if (firstKey) this.cache.delete(firstKey);
        }
    }
}
