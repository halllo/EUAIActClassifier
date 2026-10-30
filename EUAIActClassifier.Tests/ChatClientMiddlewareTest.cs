using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

/// <summary>
/// Fast, offline guards for the <see cref="IChatClient"/> middleware (<c>UseEUAIActClassification()</c> on
/// <c>ChatClientBuilder</c>): it attaches the verdict on both pipelines, forwards the model's streamed output
/// untouched, classifies once, and never throws when classification fails. These exercise the attachment logic
/// without an API key (the tier-accuracy tests in <see cref="ClassificationTest"/> remain live).
/// </summary>
[TestClass]
public sealed class ChatClientMiddlewareTest
{
    private static readonly Classification HighVerdict =
        new() { Risk = Risk.High, Category = "Annex III(4) employment", Reason = "CV screening." };

    private static ChatMessage[] Input => [new(ChatRole.User, "Screen these CVs.")];

    [TestMethod]
    public async Task NonStreaming_AttachesVerdict_AndClassifiesOnce()
    {
        var inner = new DualRoleChatClient(HighVerdict, "Here is the answer.");
        var client = inner.AsBuilder().UseEUAIActClassification().Build();

        var response = await client.GetResponseAsync(Input);

        Assert.AreEqual(Risk.High, response.EUAIActClassification?.Risk);
        Assert.AreEqual(1, inner.ClassificationCalls, "A response must be classified exactly once.");
    }

    [TestMethod]
    public async Task Streaming_ForwardsContentInOrder_AndAttachesVerdict()
    {
        var inner = new DualRoleChatClient(HighVerdict, "one ", "two ", "three");
        var client = inner.AsBuilder().UseEUAIActClassification().Build();

        var updates = await client.GetStreamingResponseAsync(Input).ToListAsync();

        // Content updates are forwarded unchanged and in order.
        var content = updates.Where(u => u.EUAIActClassification is null).Select(u => u.Text).ToList();
        CollectionAssert.AreEqual(new[] { "one ", "two ", "three" }, content);

        // The verdict is readable both off the aggregated response and off the trailing update.
        Assert.AreEqual(Risk.High, updates.ToChatResponse().EUAIActClassification?.Risk);
        Assert.AreEqual(Risk.High, updates[^1].EUAIActClassification?.Risk, "The trailing update should carry the verdict.");
        Assert.AreEqual(1, inner.ClassificationCalls, "A streaming response must be classified once.");
    }

    /// <summary>
    /// A tagged stream gets a tagged verdict: the trailing update carries the answering message's id, so
    /// <c>ToChatResponse</c> puts the verdict on that message — where a chat history store keeps it — rather than
    /// on the response.
    /// </summary>
    [TestMethod]
    public async Task Streaming_PutsTheVerdictOnTheAnsweringMessage()
    {
        var inner = new DualRoleChatClient(HighVerdict, "one ", "two") { MessageId = "msg-1" };
        var client = inner.AsBuilder().UseEUAIActClassification().Build();

        var response = (await client.GetStreamingResponseAsync(Input).ToListAsync()).ToChatResponse();

        var message = response.Messages.Single();
        Assert.AreEqual("one two", message.Text);
        Assert.AreEqual(Risk.High, message.AdditionalProperties?.EUAIActClassification?.Risk);
    }

    /// <summary>
    /// In a tool-calling loop every model call but the last ends on a tool call without a result. Such a step is not
    /// a complete turn: it costs no classification call and carries no verdict, so the loop's messages are not
    /// stamped one by one.
    /// </summary>
    [TestMethod]
    public async Task Streaming_StepEndingOnAToolCall_CarriesNoVerdict()
    {
        var inner = new DualRoleChatClient(HighVerdict, "Let me look that up.") { PrimaryContents = [new FunctionCallContent("call-1", "search_cvs")] };
        var client = inner.AsBuilder().UseEUAIActClassification().Build();

        var updates = await client.GetStreamingResponseAsync(Input).ToListAsync();

        Assert.AreEqual(0, inner.ClassificationCalls);
        Assert.IsTrue(updates.All(u => u.EUAIActClassification is null));
        Assert.IsNull(updates.ToChatResponse().EUAIActClassification);
    }

    /// <summary>The non-streaming path skips the same step and leaves its response untouched.</summary>
    [TestMethod]
    public async Task NonStreaming_StepEndingOnAToolApproval_CarriesNoVerdict()
    {
        var inner = new DualRoleChatClient(HighVerdict) { PrimaryContents = [new ToolApprovalRequestContent("approval-1", new FunctionCallContent("call-1", "screen_cvs"))] };
        var client = inner.AsBuilder().UseEUAIActClassification().Build();

        var response = await client.GetResponseAsync(Input);

        Assert.AreEqual(0, inner.ClassificationCalls);
        Assert.IsNull(response.EUAIActClassification);
    }

    /// <summary>The model call after the tool ran sees the call answered, and that completes the turn.</summary>
    [TestMethod]
    public async Task Streaming_AnswerAfterTheToolRan_IsClassified()
    {
        var inner = new DualRoleChatClient(HighVerdict, "Here is the ranking.");
        var client = inner.AsBuilder().UseEUAIActClassification().Build();

        var updates = await client.GetStreamingResponseAsync(
        [
            new ChatMessage(ChatRole.User, "Screen these CVs."),
            new ChatMessage(ChatRole.Assistant, [new FunctionCallContent("call-1", "search_cvs")]),
            new ChatMessage(ChatRole.Tool, [new FunctionResultContent("call-1", "3 CVs")]),
        ]).ToListAsync();

        Assert.AreEqual(1, inner.ClassificationCalls);
        Assert.AreEqual(Risk.High, updates[^1].EUAIActClassification?.Risk);
    }

    [TestMethod]
    public async Task ClassifierThrows_RecordsUnknown_DoesNotThrow()
    {
        var inner = new DualRoleChatClient(new InvalidOperationException("boom"), "Here is the answer.");
        var client = inner.AsBuilder().UseEUAIActClassification().Build();

        var response = await client.GetResponseAsync(Input); // must not throw

        Assert.AreEqual(Risk.Unknown, response.EUAIActClassification?.Risk);
        StringAssert.Contains(response.EUAIActClassification?.Reason, "boom");
    }

    [TestMethod]
    public async Task EmptyConversation_ReturnsUnknown_WithoutCallingClassifier()
    {
        // The verdict is intentionally unreachable: the empty-conversation guard short-circuits before any model call.
        var classifier = new RecordingClassifier(HighVerdict);

        var verdict = await classifier.ClassifyEUAIActRiskAsync([new(ChatRole.User, "   ")]);

        Assert.AreEqual(Risk.Unknown, verdict.Risk);
        Assert.AreEqual(0, classifier.Calls, "There is nothing to classify, so the model must not be called.");
    }

    [TestMethod]
    public async Task Middleware_EmptyConversation_ReturnsUnknown_WithoutClassifying()
    {
        // No primary content + whitespace-only input → nothing to classify, exercised through the full pipeline.
        var inner = new DualRoleChatClient(HighVerdict);
        var client = inner.AsBuilder().UseEUAIActClassification().Build();

        var response = await client.GetResponseAsync([new(ChatRole.User, "   ")]);

        Assert.AreEqual(Risk.Unknown, response.EUAIActClassification?.Risk);
        Assert.AreEqual(0, inner.ClassificationCalls, "Nothing to classify, so the model must not be called.");
    }

    [TestMethod]
    public async Task DefaultOptions_UsesBuiltInSystemPrompt()
    {
        var classifier = new RecordingClassifier(HighVerdict);

        await classifier.ClassifyEUAIActRiskAsync([new(ChatRole.User, "Screen these CVs.")]);

        var system = classifier.LastMessages!.Single(m => m.Role == ChatRole.System);
        StringAssert.Contains(system.Text, "EU AI Act compliance classifier", "The built-in prompt should be used by default.");
    }

    [TestMethod]
    public async Task CustomSystemPrompt_ReplacesBuiltInPrompt()
    {
        var classifier = new RecordingClassifier(HighVerdict);

        await classifier.ClassifyEUAIActRiskAsync(
            [new(ChatRole.User, "Screen these CVs.")],
            new ClassificationOptions { SystemPrompt = "CUSTOM PROMPT" });

        var system = classifier.LastMessages!.Single(m => m.Role == ChatRole.System);
        Assert.AreEqual("CUSTOM PROMPT", system.Text, "A custom prompt should fully replace the built-in one.");
    }

    [TestMethod]
    public async Task ConversationFilter_RestrictsClassifiedMessages()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        ChatMessage[] conversation =
        [
            new(ChatRole.User, "OLDEST question"),
            new(ChatRole.Assistant, "old answer"),
            new(ChatRole.User, "NEWEST question"),
            new(ChatRole.Assistant, "new answer"),
        ];

        // Keep only the last turn (a user message and the assistant reply).
        await classifier.ClassifyEUAIActRiskAsync(
            conversation,
            new ClassificationOptions { ConversationFilter = m => m.TakeLast(2) });

        var transcript = classifier.LastMessages!.Single(m => m.Role == ChatRole.User).Text;
        StringAssert.Contains(transcript, "NEWEST question");
        StringAssert.Contains(transcript, "new answer");
        Assert.IsFalse(transcript.Contains("OLDEST"), "Filtered-out turns must not reach the classifier.");
    }

    [TestMethod]
    public async Task DefaultOptions_ClassifiesAtTemperatureZero()
    {
        var classifier = new RecordingClassifier(HighVerdict);

        await classifier.ClassifyEUAIActRiskAsync([new(ChatRole.User, "Screen these CVs.")]);

        Assert.AreEqual(0.0F, classifier.LastOptions?.Temperature, "The classifier should default to temperature 0.");
    }

    [TestMethod]
    public async Task CustomTemperature_IsPassedToTheClassificationCall()
    {
        var classifier = new RecordingClassifier(HighVerdict);

        await classifier.ClassifyEUAIActRiskAsync(
            [new(ChatRole.User, "Screen these CVs.")],
            new ClassificationOptions { Temperature = 0.7F });

        Assert.AreEqual(0.7F, classifier.LastOptions?.Temperature);
    }

    [TestMethod]
    public async Task ModelDefaultTemperature_SendsNoTemperature()
    {
        var classifier = new RecordingClassifier(HighVerdict);

        await classifier.ClassifyEUAIActRiskAsync(
            [new(ChatRole.User, "Screen these CVs.")],
            new ClassificationOptions { Temperature = ClassificationOptions.ModelDefaultTemperature });

        Assert.IsNull(classifier.LastOptions?.Temperature, "The opt-out sentinel must leave Temperature unset.");
    }

    [TestMethod]
    public async Task Middleware_PassesOptionsThrough_PromptFilterAndTemperature()
    {
        var inner = new DualRoleChatClient(HighVerdict, "Here is the answer.");
        var options = new ClassificationOptions
        {
            SystemPrompt = "CUSTOM",
            ConversationFilter = m => m.TakeLast(1), // of the full conversation (input + reply) → just the reply
            Temperature = 0.3F,
        };
        var client = inner.AsBuilder().UseEUAIActClassification(options).Build();

        await client.GetResponseAsync([new(ChatRole.User, "FIRST"), new(ChatRole.User, "SECOND")]);

        Assert.AreEqual("CUSTOM", inner.LastClassificationMessages!.Single(m => m.Role == ChatRole.System).Text);
        var transcript = inner.LastClassificationMessages!.Single(m => m.Role == ChatRole.User).Text;
        StringAssert.Contains(transcript, "Here is the answer.");
        Assert.IsFalse(transcript.Contains("FIRST"), "The filter should have dropped earlier input messages.");
        Assert.AreEqual(0.3F, inner.LastClassificationOptions?.Temperature);
    }
}
