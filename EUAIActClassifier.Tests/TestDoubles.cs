using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

/// <summary>An inner agent that streams a fixed set of assistant messages (one update each).</summary>
internal sealed class StubAgent : AIAgent
{
    private readonly IReadOnlyList<ChatMessage> _messages;

    /// <summary>An agent whose run is a plain answer: one assistant text chunk per update.</summary>
    public StubAgent(params string[] chunks)
        => _messages = [.. chunks.Select(c => new ChatMessage(ChatRole.Assistant, c))];

    private StubAgent(IReadOnlyList<ChatMessage> messages) => _messages = messages;

    /// <summary>
    /// An agent whose run ends on these contents rather than on text — how an interrupted run looks, where the last
    /// thing produced is a <see cref="ToolApprovalRequestContent"/> waiting for a decision and no answer exists.
    /// </summary>
    public static StubAgent Producing(params AIContent[] contents)
        => new([new ChatMessage(ChatRole.Assistant, [.. contents])]);

    protected override Task<AgentResponse> RunCoreAsync(
        IEnumerable<ChatMessage> messages, AgentSession? session, AgentRunOptions? options, CancellationToken cancellationToken)
        => Task.FromResult(new AgentResponse([.. _messages]));

    protected override async IAsyncEnumerable<AgentResponseUpdate> RunCoreStreamingAsync(
        IEnumerable<ChatMessage> messages, AgentSession? session, AgentRunOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        foreach (var message in _messages)
        {
            yield return new AgentResponseUpdate(ChatRole.Assistant, message.Contents);
        }
        await Task.CompletedTask;
    }

    // Session persistence is unused by these tests.
    protected override ValueTask<AgentSession> CreateSessionCoreAsync(CancellationToken cancellationToken)
        => throw new NotSupportedException();

    protected override ValueTask<JsonElement> SerializeSessionCoreAsync(AgentSession session, JsonSerializerOptions? jsonSerializerOptions, CancellationToken cancellationToken)
        => throw new NotSupportedException();

    protected override ValueTask<AgentSession> DeserializeSessionCoreAsync(JsonElement serializedSession, JsonSerializerOptions? jsonSerializerOptions, CancellationToken cancellationToken)
        => throw new NotSupportedException();
}

/// <summary>
/// A chat client used purely as the classification engine: it answers the structured-output classification call
/// with a fixed verdict (as JSON), or throws to exercise the best-effort failure path. Counts invocations so a
/// test can prove classification happens exactly once per run.
/// </summary>
internal sealed class RecordingClassifier : IChatClient
{
    private readonly Classification? _verdict;
    private readonly Exception? _throw;

    public RecordingClassifier(Classification verdict) => _verdict = verdict;
    public RecordingClassifier(Exception toThrow) => _throw = toThrow;

    public int Calls { get; private set; }

    /// <summary>The messages received on the most recent classification call (system prompt + transcript).</summary>
    public IReadOnlyList<ChatMessage>? LastMessages { get; private set; }

    /// <summary>The <see cref="ChatOptions"/> received on the most recent classification call.</summary>
    public ChatOptions? LastOptions { get; private set; }

    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        Calls++;
        LastMessages = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();
        LastOptions = options;
        if (_throw is not null) throw _throw;
        // GetResponseAsync<Classification> parses the assistant message text back into a Classification.
        return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, JsonSerializer.Serialize(_verdict))));
    }

    // ClassifyEUAIActRiskAsync only ever uses the non-streaming GetResponseAsync<T> overload.
    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException();

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose() { }
}

/// <summary>
/// A single inner chat client that plays both roles for the <see cref="IChatClient"/> middleware: it returns the
/// primary model output for normal calls, and a fixed classification verdict (or throws) for the structured-output
/// classification call. The two are told apart by the JSON response format that <c>GetResponseAsync&lt;T&gt;</c>
/// sets on the classification request.
/// </summary>
internal sealed class DualRoleChatClient : IChatClient
{
    private readonly IReadOnlyList<string> _primaryChunks;
    private readonly Classification? _verdict;
    private readonly Exception? _classifierThrow;

    public DualRoleChatClient(Classification verdict, params string[] primaryChunks)
    {
        _verdict = verdict;
        _primaryChunks = primaryChunks;
    }

    public DualRoleChatClient(Exception classifierThrow, params string[] primaryChunks)
    {
        _classifierThrow = classifierThrow;
        _primaryChunks = primaryChunks;
    }

    /// <summary>
    /// Contents the primary turn ends on instead of text — a tool call the run hands on, or one awaiting approval —
    /// streamed as one update after the text chunks.
    /// </summary>
    public IReadOnlyList<AIContent> PrimaryContents { get; init; } = [];

    /// <summary>The message id the primary turn's updates carry, as a provider that names its messages sets it.</summary>
    public string? MessageId { get; init; }

    public int ClassificationCalls { get; private set; }

    /// <summary>The messages received on the most recent classification call (system prompt + transcript).</summary>
    public IReadOnlyList<ChatMessage>? LastClassificationMessages { get; private set; }

    /// <summary>The <see cref="ChatOptions"/> received on the most recent classification call.</summary>
    public ChatOptions? LastClassificationOptions { get; private set; }

    private static bool IsClassificationCall(ChatOptions? options) => options?.ResponseFormat is ChatResponseFormatJson;

    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (IsClassificationCall(options))
        {
            ClassificationCalls++;
            LastClassificationMessages = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();
            LastClassificationOptions = options;
            if (_classifierThrow is not null) throw _classifierThrow;
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, JsonSerializer.Serialize(_verdict))));
        }

        List<ChatMessage> primary = [.. _primaryChunks.Select(c => new ChatMessage(ChatRole.Assistant, c) { MessageId = MessageId })];
        if (PrimaryContents.Count > 0) primary.Add(new ChatMessage(ChatRole.Assistant, [.. PrimaryContents]) { MessageId = MessageId });
        return Task.FromResult(new ChatResponse(primary));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Only the primary turn streams; the classification call always uses the non-streaming overload.
        foreach (var c in _primaryChunks)
        {
            yield return new ChatResponseUpdate(ChatRole.Assistant, c) { MessageId = MessageId };
        }
        if (PrimaryContents.Count > 0)
        {
            yield return new ChatResponseUpdate(ChatRole.Assistant, [.. PrimaryContents]) { MessageId = MessageId };
        }
        await Task.CompletedTask;
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose() { }
}
