using System.Runtime.CompilerServices;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

/// <summary>
/// A Microsoft Agent Framework agent that wraps an inner <see cref="AIAgent"/> and classifies the EU AI Act risk
/// tier of the completed turn <b>once per agent run</b>, attaching the verdict so it can be read back via
/// <c>response.EUAIActClassification</c> or <c>update.EUAIActClassification</c>.
/// </summary>
/// <remarks>
/// Unlike placing <see cref="ClassificationMiddleware.UseEUAIActClassification(ChatClientBuilder)"/> in the agent's
/// <see cref="IChatClient"/> pipeline — which an agent run loop invokes once per model round-trip, re-classifying
/// every intermediate tool-calling step — this agent sits at the run layer and classifies exactly once, over the
/// full conversation. The supplied <c>classifier</c> is used purely as the classification engine and is not part of
/// the agent's inference path. Classification is a best-effort side channel: it never throws into the primary run and
/// records <see cref="Risk.Unknown"/> on failure.
/// <para>
/// Use this <b>instead of</b>, not in addition to, the <see cref="IChatClient"/>-level middleware on the same agent:
/// installing both classifies every turn twice (once per model call, then once per run).
/// </para>
/// </remarks>
public sealed class EUAIActClassificationAgent : DelegatingAIAgent
{
    private readonly IChatClient _classifier;

    /// <summary>Initializes a new instance of the <see cref="EUAIActClassificationAgent"/> class.</summary>
    /// <param name="innerAgent">The agent whose runs are classified.</param>
    /// <param name="classifier">The chat client used as the classification engine (not part of the inference path).</param>
    public EUAIActClassificationAgent(AIAgent innerAgent, IChatClient classifier)
        : base(innerAgent)
        => _classifier = classifier ?? throw new ArgumentNullException(nameof(classifier));

    /// <inheritdoc />
    protected override async Task<AgentResponse> RunCoreAsync(
        IEnumerable<ChatMessage> messages,
        AgentSession? session,
        AgentRunOptions? options,
        CancellationToken cancellationToken)
    {
        var messageList = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();

        var response = await InnerAgent.RunAsync(messageList, session, options, cancellationToken).ConfigureAwait(false);

        var classification = await _classifier
            .ClassifyEUAIActRiskAsync(messageList.Concat(response.Messages), cancellationToken)
            .ConfigureAwait(false);

        (response.AdditionalProperties ??= [])[ClassificationMiddleware.ClassificationKey] = classification;
        return response;
    }

    /// <inheritdoc />
    protected override async IAsyncEnumerable<AgentResponseUpdate> RunCoreStreamingAsync(
        IEnumerable<ChatMessage> messages,
        AgentSession? session,
        AgentRunOptions? options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var messageList = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();

        // Forward every update untouched so the consumer's stream is never blocked by classification.
        List<AgentResponseUpdate> updates = [];
        await foreach (var update in InnerAgent.RunStreamingAsync(messageList, session, options, cancellationToken)
                           .ConfigureAwait(false))
        {
            updates.Add(update);
            yield return update;
        }

        // Once the run has completed, classify the full conversation exactly once and emit the verdict as a
        // trailing side-channel update.
        var classification = await _classifier
            .ClassifyEUAIActRiskAsync(messageList.Concat(updates.ToAgentResponse().Messages), cancellationToken)
            .ConfigureAwait(false);

        yield return new AgentResponseUpdate
        {
            AdditionalProperties = new() { [ClassificationMiddleware.ClassificationKey] = classification },
        };
    }
}

/// <summary>Extension methods for adding and reading EU AI Act classification on a Microsoft Agent Framework agent.</summary>
public static class ClassificationAgentExtensions
{
    extension(AIAgent agent)
    {
        /// <summary>
        /// Wraps this agent so that every run is classified against the EU AI Act risk tiers exactly once, over the
        /// completed conversation. Composes with <c>AsBuilder()</c>: <c>agent.AsBuilder().Use(a =&gt;
        /// a.UseEUAIActClassification(classifier)).Build()</c>, or use the returned agent directly.
        /// </summary>
        /// <param name="classifier">The chat client used as the classification engine (not part of the inference path).</param>
        /// <returns>An <see cref="AIAgent"/> that classifies each run.</returns>
        public AIAgent UseEUAIActClassification(IChatClient classifier) =>
            new EUAIActClassificationAgent(agent, classifier);
    }

    extension(AgentResponse response)
    {
        /// <summary>
        /// Gets the EU AI Act classification attached to this agent response, or <see langword="null"/> if none was
        /// attached (for example when the agent was not wrapped with <c>UseEUAIActClassification</c>).
        /// </summary>
        /// <remarks>
        /// Both the response level and the last message are checked, mirroring the <see cref="IChatClient"/> side:
        /// depending on how a downstream layer coalesces the trailing side-channel update, the verdict can land in
        /// either location.
        /// </remarks>
        public Classification? EUAIActClassification =>
            ClassificationMiddleware.GetClassification(response.AdditionalProperties)
            ?? ClassificationMiddleware.GetClassification(response.Messages.LastOrDefault()?.AdditionalProperties);
    }

    extension(AgentResponseUpdate update)
    {
        /// <summary>
        /// Gets the EU AI Act classification carried by this streaming agent update, or <see langword="null"/> if none.
        /// </summary>
        /// <remarks>
        /// The verdict rides on a trailing side-channel update emitted after the run completes; intermediate content
        /// updates return <see langword="null"/>.
        /// </remarks>
        public Classification? EUAIActClassification =>
            ClassificationMiddleware.GetClassification(update.AdditionalProperties);
    }
}
