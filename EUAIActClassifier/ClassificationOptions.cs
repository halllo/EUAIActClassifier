using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

/// <summary>
/// Optional configuration for EU AI Act classification: override the system prompt used by the classifier
/// and/or restrict which messages of a conversation are sent for classification.
/// </summary>
public sealed class ClassificationOptions
{
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
