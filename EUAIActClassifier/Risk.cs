using System.Text.Json.Serialization;
using DescriptionAttribute = System.ComponentModel.DescriptionAttribute;

namespace EUAIActClassifier;

/// <summary>
/// The risk tier an AI use case falls into under the EU AI Act's risk-based approach.
/// Ordered by increasing severity so that values can be compared (<c>&gt;=</c>) against a threshold.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<Risk>))]
public enum Risk
{
    /// <summary>The risk tier could not be determined from the conversation.</summary>
    [Description("The risk tier could not be determined.")]
    Unknown = 0,

    /// <summary>
    /// Minimal or no risk. The default tier for the vast majority of AI uses; the EU AI Act imposes no
    /// obligations. Examples: general questions and chit-chat, summarisation, translation, coding help,
    /// spam filters, AI in video games.
    /// </summary>
    [Description("Minimal or no risk: no EU AI Act obligations (e.g. general Q&A, summarisation, coding help, spam filters, AI in games).")]
    Minimal,

    /// <summary>
    /// Limited / transparency risk (Article 50). The use case is allowed but carries disclosure duties:
    /// AI that interacts with people as a chatbot, or that generates synthetic or "deepfake"
    /// audio/image/video/text. Users must be told they are dealing with AI or with artificially generated
    /// content.
    /// </summary>
    [Description("Limited / transparency risk (Art. 50): chatbots that talk to people, deepfakes / synthetic media. Allowed but must disclose.")]
    Limited,

    /// <summary>
    /// High risk (Article 6 and Annex III, or a safety component of a regulated product under Annex I).
    /// Use cases in sensitive domains: remote biometric identification, biometric categorisation and emotion
    /// recognition, critical infrastructure, education and exam grading, employment and recruitment, access to
    /// essential services such as credit scoring or insurance, law enforcement, migration and border control,
    /// administration of justice and democratic processes.
    /// </summary>
    [Description("High risk (Art. 6 / Annex III): remote biometrics, emotion recognition, critical infrastructure, education grading, hiring, credit scoring, insurance, law enforcement, migration, justice. Heavy compliance obligations.")]
    High,

    /// <summary>
    /// Unacceptable risk: a prohibited practice under Article 5. Examples: social scoring, subliminal or
    /// manipulative techniques, exploiting vulnerabilities, untargeted scraping of facial images to build
    /// recognition databases, emotion recognition in the workplace or schools, biometric categorisation by
    /// sensitive traits, predictive policing from profiling, and real-time remote biometric identification in
    /// public spaces. These systems are banned outright.
    /// </summary>
    [Description("Unacceptable risk (Art. 5): prohibited practices such as social scoring, manipulation, untargeted facial-image scraping, predictive policing by profiling. Banned outright.")]
    Unacceptable,
}
