using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

/// <summary>
/// Optional configuration for EU AI Act classification: override the system prompt used by the classifier,
/// restrict which messages of a conversation are sent for classification, and/or change the sampling
/// temperature of the classification call.
/// </summary>
public sealed class ClassificationOptions
{
    /// <summary>The sampling temperature used for the classification call when none is configured.</summary>
    internal const float DefaultTemperature = 0.0F;

    /// <summary>
    /// A <see cref="Temperature"/> value that sends no temperature at all, leaving the model's own default in
    /// place — for endpoints and reasoning models that reject the parameter.
    /// </summary>
    public const float ModelDefaultTemperature = float.NaN;

    /// <summary>
    /// The sampling temperature for the classification call. When <see langword="null"/>, the classifier runs at
    /// <c>0.0</c> — deterministic verdicts for the same conversation.
    /// </summary>
    /// <remarks>
    /// Raise this only if a model rejects or degrades at temperature 0; a higher temperature makes the tier
    /// assigned to one and the same conversation vary between runs. Set it to
    /// <see cref="ModelDefaultTemperature"/> to send no temperature at all and let the model's own default apply.
    /// </remarks>
    public float? Temperature { get; set; }

    /// <summary>
    /// A custom system prompt that <b>fully replaces</b> the built-in EU AI Act classification prompt.
    /// When <see langword="null"/> or whitespace, the built-in prompt is used.
    /// </summary>
    /// <remarks>
    /// The built-in prompt encodes all four risk-tier definitions and their legal bases (Article 5, Annex III,
    /// Article 50, …). A replacement takes full responsibility for instructing the model to return a
    /// <see cref="Classification"/> (its Risk / Category / Reason fields).
    /// </remarks>
    public string? SystemPrompt { get; set; }

    /// <summary>
    /// An optional filter applied to the full conversation before it is classified — for example, to classify
    /// only the most recent turns and cut tokens. It receives the complete conversation (the request together
    /// with the response messages) and returns the subset to classify. When <see langword="null"/>, the whole
    /// conversation is classified.
    /// </summary>
    /// <remarks>
    /// Example — classify only the last two turns (a user message and the assistant reply each):
    /// <code>options.ConversationFilter = messages => messages.TakeLast(4);</code>
    /// The filter runs first; empty-text messages are still dropped afterwards.
    /// </remarks>
    public Func<IReadOnlyList<ChatMessage>, IEnumerable<ChatMessage>>? ConversationFilter { get; set; }
}
