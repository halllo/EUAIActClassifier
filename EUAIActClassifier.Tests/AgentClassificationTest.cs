using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

/// <summary>
/// Fast, offline guards for the agent-framework middleware (<see cref="EUAIActClassificationAgent"/>): it classifies
/// each run exactly once over the completed conversation, forwards the inner agent's output untouched, exposes the
/// verdict via the agent-side getters, and never throws when classification fails. No live model required.
/// </summary>
[TestClass]
public sealed class AgentClassificationTest
{
    private static readonly Classification HighVerdict =
        new() { Risk = Risk.High, Category = "Annex III(4) employment", Reason = "CV screening." };

    private static ChatMessage[] Input => [new(ChatRole.User, "Screen these CVs.")];

    [TestMethod]
    public async Task RunAsync_AttachesVerdict_AndClassifiesOnce()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        var agent = new StubAgent("Here is the answer.").UseEUAIActClassification(classifier);

        var response = await agent.RunAsync(Input);

        Assert.AreEqual(Risk.High, response.EUAIActClassification?.Risk, "The aggregated response should carry the verdict.");
        Assert.AreEqual("Annex III(4) employment", response.EUAIActClassification?.Category);
        Assert.AreEqual(1, classifier.Calls, "A run must classify exactly once.");
    }

    [TestMethod]
    public async Task RunStreamingAsync_ForwardsContentInOrder_AndTrailingVerdictOnce()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        var agent = new StubAgent("one ", "two ", "three").UseEUAIActClassification(classifier);

        var updates = new List<AgentResponseUpdate>();
        await foreach (var update in agent.RunStreamingAsync(Input))
        {
            updates.Add(update);
        }

        // Inner content updates are forwarded unchanged and in order.
        var content = updates.Where(u => u.EUAIActClassification is null).Select(u => u.Text).ToList();
        CollectionAssert.AreEqual(new[] { "one ", "two ", "three" }, content, "Inner updates must be forwarded unchanged and in order.");

        // Exactly one trailing side-channel update carries the verdict, and it is last.
        var carriers = updates.Where(u => u.EUAIActClassification is not null).ToList();
        Assert.AreEqual(1, carriers.Count, "Exactly one streamed update should carry the verdict.");
        Assert.AreSame(updates[^1], carriers[0], "The verdict must ride on the trailing update.");
        Assert.AreEqual(Risk.High, carriers[0].EUAIActClassification?.Risk);
        Assert.AreEqual(1, classifier.Calls, "A streaming run must classify once, not per update.");
    }

    [TestMethod]
    public async Task RunAsync_ClassifierThrows_RecordsUnknown_DoesNotThrow()
    {
        var agent = new StubAgent("ok").UseEUAIActClassification(new RecordingClassifier(new InvalidOperationException("boom")));

        var response = await agent.RunAsync(Input); // must not throw

        Assert.AreEqual(Risk.Unknown, response.EUAIActClassification?.Risk);
        StringAssert.Contains(response.EUAIActClassification?.Reason, "boom", "The failure reason should be surfaced.");
    }

    [TestMethod]
    public async Task RunAsync_PassesOptionsThrough_PromptFilterAndTemperature()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        var options = new ClassificationOptions
        {
            SystemPrompt = "CUSTOM",
            ConversationFilter = m => m.TakeLast(1), // of the full conversation (input + reply) → just the reply
            Temperature = 0.3F,
        };
        var agent = new StubAgent("Here is the answer.").UseEUAIActClassification(classifier, options);

        await agent.RunAsync([new(ChatRole.User, "Screen these CVs.")]);

        Assert.AreEqual("CUSTOM", classifier.LastMessages!.Single(m => m.Role == ChatRole.System).Text);
        var transcript = classifier.LastMessages!.Single(m => m.Role == ChatRole.User).Text;
        StringAssert.Contains(transcript, "Here is the answer.");
        Assert.IsFalse(transcript.Contains("Screen these CVs."), "The filter should have dropped the earlier input message.");
        Assert.AreEqual(0.3F, classifier.LastOptions?.Temperature);
    }

    /// <summary>
    /// A run that stops to ask for a tool approval has not completed its turn: the tool has not run and there is no
    /// answer, so it is not classified — no classification call is spent, and the verdict stays
    /// <see cref="Risk.Unknown"/> so a host that surfaces High-and-above sees nothing for the round-trip.
    /// </summary>
    [TestMethod]
    public async Task RunAsync_RunStopsForToolApproval_DoesNotClassify()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        var pending = new ToolApprovalRequestContent("approval-1", new FunctionCallContent("call-1", "screen_cvs"));
        var agent = StubAgent.Producing(pending).UseEUAIActClassification(classifier);

        var response = await agent.RunAsync(Input);

        Assert.AreEqual(0, classifier.Calls, "An interrupted run must not spend a classification call.");
        Assert.AreEqual(Risk.Unknown, response.EUAIActClassification?.Risk);
        StringAssert.Contains(response.EUAIActClassification?.Reason, "approval", "The reason should name why it was skipped.");
    }

    /// <summary>The streaming path skips the same run, and still emits its trailing verdict update.</summary>
    [TestMethod]
    public async Task RunStreamingAsync_RunStopsForToolApproval_DoesNotClassify()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        var pending = new ToolApprovalRequestContent("approval-1", new FunctionCallContent("call-1", "screen_cvs"));
        var agent = StubAgent.Producing(pending).UseEUAIActClassification(classifier);

        var updates = new List<AgentResponseUpdate>();
        await foreach (var update in agent.RunStreamingAsync(Input))
        {
            updates.Add(update);
        }

        Assert.AreEqual(0, classifier.Calls, "An interrupted streaming run must not spend a classification call.");
        Assert.AreEqual(Risk.Unknown, updates[^1].EUAIActClassification?.Risk);
    }

    /// <summary>
    /// The other half of the rule: once the decision is in, the turn completes and is classified as usual. It is the
    /// request being <i>unanswered</i> that suspends the verdict, not its presence — otherwise the resume run, which
    /// carries the answered request, would never be classified either.
    /// </summary>
    [TestMethod]
    public async Task RunAsync_ApprovalAnswered_ClassifiesTheCompletedTurn()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        var agent = new StubAgent("I have screened the applicants.").UseEUAIActClassification(classifier);
        var call = new FunctionCallContent("call-1", "screen_cvs");

        var response = await agent.RunAsync(
        [
            new ChatMessage(ChatRole.User, "Screen these CVs."),
            new ChatMessage(ChatRole.Assistant, [new ToolApprovalRequestContent("approval-1", call)]),
            new ChatMessage(ChatRole.User, [new ToolApprovalResponseContent("approval-1", true, call)]),
        ]);

        Assert.AreEqual(1, classifier.Calls, "An answered approval must not suspend the verdict.");
        Assert.AreEqual(Risk.High, response.EUAIActClassification?.Risk);
    }

    /// <summary>
    /// A run that ends by handing a tool call to its caller — a client-side tool the host executes — has not
    /// completed its turn either: the call has no result and there is no answer yet, so no classification is spent.
    /// </summary>
    [TestMethod]
    public async Task RunStreamingAsync_RunEndsOnUnansweredToolCall_DoesNotClassify()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        var agent = StubAgent.Producing(new TextContent("Changing the colour first."), new FunctionCallContent("call-1", "change_background_color"))
            .UseEUAIActClassification(classifier);

        var updates = new List<AgentResponseUpdate>();
        await foreach (var update in agent.RunStreamingAsync(Input))
        {
            updates.Add(update);
        }

        Assert.AreEqual(0, classifier.Calls, "A run waiting on a tool result must not spend a classification call.");
        Assert.AreEqual(Risk.Unknown, updates[^1].EUAIActClassification?.Risk);
        StringAssert.Contains(updates[^1].EUAIActClassification?.Reason, "tool result", "The reason should name why it was skipped.");
    }

    /// <summary>
    /// The run that carries the result back and answers completes the turn, so it is classified. Its input holds
    /// only the result: the call it answers lives in the history, not in this run's messages.
    /// </summary>
    [TestMethod]
    public async Task RunAsync_ToolResultSentBack_ClassifiesTheCompletedTurn()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        var agent = new StubAgent("The colour is changed; here is the shortlist.").UseEUAIActClassification(classifier);

        var response = await agent.RunAsync(
        [
            new ChatMessage(ChatRole.Tool, [new FunctionResultContent("call-1", "done")]),
        ]);

        Assert.AreEqual(1, classifier.Calls);
        Assert.AreEqual(Risk.High, response.EUAIActClassification?.Risk);
    }

    /// <summary>A tool the run executed itself carries its result in the same run, so the turn is complete.</summary>
    [TestMethod]
    public async Task RunAsync_ToolCallAnsweredWithinTheRun_Classifies()
    {
        var classifier = new RecordingClassifier(HighVerdict);
        var agent = StubAgent.Producing(
                new FunctionCallContent("call-1", "search_cvs"),
                new FunctionResultContent("call-1", "3 CVs"),
                new TextContent("I have ranked the applicants."))
            .UseEUAIActClassification(classifier);

        var response = await agent.RunAsync(Input);

        Assert.AreEqual(1, classifier.Calls);
        Assert.AreEqual(Risk.High, response.EUAIActClassification?.Risk);
    }

    [TestMethod]
    public async Task RunStreamingAsync_ClassifierThrows_RecordsUnknown_DoesNotThrow()
    {
        var agent = new StubAgent("ok").UseEUAIActClassification(new RecordingClassifier(new OperationCanceledException()));

        var updates = new List<AgentResponseUpdate>();
        await foreach (var update in agent.RunStreamingAsync(Input)) // must not throw
        {
            updates.Add(update);
        }

        var verdict = updates[^1].EUAIActClassification;
        Assert.AreEqual(Risk.Unknown, verdict?.Risk);
        StringAssert.Contains(verdict?.Reason, "cancel", "Cancellation should map to the cancelled-reason path.");
    }
}
