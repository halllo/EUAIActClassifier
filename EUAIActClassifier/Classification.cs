using DescriptionAttribute = System.ComponentModel.DescriptionAttribute;

namespace EUAIActClassifier;

/// <summary>
/// The EU AI Act classification inferred for a chat conversation: which risk tier the use case falls into,
/// the concrete legal basis, and a short justification.
/// </summary>
public class Classification
{
    /// <summary>The EU AI Act risk tier the use case falls into.</summary>
    [Description("The EU AI Act risk tier the use case falls into.")]
    public Risk Risk { get; set; }

    /// <summary>
    /// The concrete EU AI Act basis for the classification, e.g. "Annex III(4) employment",
    /// "Article 5(1)(c) social scoring", "Article 50 transparency", or "Minimal risk – general assistance".
    /// </summary>
    [Description("The concrete EU AI Act basis, e.g. 'Annex III(4) employment', 'Article 5(1)(c) social scoring', 'Article 50 transparency', or 'Minimal risk'.")]
    public string Category { get; set; } = "";

    /// <summary>A short, factual justification (one or two sentences) for the assigned risk tier.</summary>
    [Description("A short, factual justification (one or two sentences) for the assigned risk tier.")]
    public string Reason { get; set; } = "";
}
