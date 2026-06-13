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
