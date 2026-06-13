using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

/// <summary>
/// Middleware for <see cref="IChatClient"/> that classifies every conversation against the EU AI Act risk
/// tiers and attaches the result so it can be read back via <see cref="EUAIActClassification"/>.
/// </summary>
/// <remarks>
/// Classification is a best-effort side channel: it never throws into the primary chat response. If the
/// classification call fails it is recorded as <see cref="Risk.Unknown"/> instead.
/// </remarks>
public static class ClassificationMiddleware
{
    /// <summary>The key under which the classification is stored in an <see cref="AdditionalPropertiesDictionary"/>.</summary>
    private const string ClassificationKey = "EUAIActClassifier.Classification";

    /// <summary>
    /// System prompt instructing the model to classify a conversation into one of the EU AI Act risk tiers.
    /// The tier definitions mirror Article 5 (prohibited), Article 6 / Annex III (high risk) and Article 50
    /// (transparency / limited risk); everything else is minimal risk.
    /// </summary>
    private const string SystemPrompt =
        """
        You are an EU AI Act compliance classifier. Given a conversation between a user and an AI assistant,
        determine which EU AI Act risk tier best describes the AI USE CASE revealed by the conversation, and
        name the concrete legal basis.

        Classify the intended PURPOSE / USE CASE of the AI system, not whether the topic merely sounds
        sensitive or dangerous. A request can discuss a dangerous topic yet still be Minimal risk when the AI
        only acts as a general information assistant. Assign the HIGHEST tier that genuinely applies. You judge
        only the use-case risk tier; obligations for general-purpose AI models themselves (Chapter V) are out
        of scope. Never answer Unknown — if you are unsure, choose the closest tier (Minimal by default).

        Tiers, from lowest to highest severity:

        MINIMAL — the default. No EU AI Act obligations. General knowledge questions, chit-chat, summarisation,
        translation, brainstorming, coding help, spam filtering, recommendations, AI in video games, and
        biometric verification/authentication that only confirms a person is who they claim to be (e.g.
        unlocking a phone). If nothing more specific applies, choose Minimal.

        LIMITED — transparency obligations under Article 50. The use case is one of:
        (a) a chatbot / conversational agent meant to interact with people, who must be told they are talking to AI;
        (b) generating synthetic or "deepfake" audio, image, video or text intended to look real, which must be
            marked as artificially generated.
        Such systems are allowed, but the provider or deployer must disclose the AI or synthetic nature.

        HIGH — Article 6 and Annex III, or an AI safety component of a regulated product (Annex I). The use case
        operates in a sensitive domain and materially affects people's rights or safety:
        - Remote biometric identification; biometric categorisation by non-sensitive attributes; emotion
          recognition (where not prohibited). These also carry an Article 50(3) transparency duty.
        - Critical infrastructure: safety components for road traffic, water, gas, heating, electricity, digital infrastructure.
        - Education & vocational training: admissions, grading exams, evaluating learning outcomes, exam proctoring.
        - Employment: recruiting, CV screening or ranking, hiring/firing decisions, task allocation, performance monitoring.
        - Access to essential services: creditworthiness / credit scoring (except fraud detection), eligibility for public benefits, life/health insurance risk or pricing, emergency call triage and dispatch.
        - Law enforcement: assessing the risk of offending or of becoming a victim, evidence reliability, profiling during investigations; post (retrospective) remote biometric identification.
        - Migration, asylum & border control: risk assessment, examining visa or asylum applications, identification.
        - Administration of justice and democratic processes: assisting judicial decision-making, influencing elections or voting behaviour.
        - Safety components of regulated products such as machinery, medical devices, vehicles or toys.

        UNACCEPTABLE — a prohibited practice under Article 5. Banned outright:
        - Subliminal, manipulative or deceptive techniques that materially distort behaviour and cause harm.
        - Exploiting vulnerabilities of age, disability or socio-economic situation to distort behaviour and cause harm.
        - Social scoring: evaluating or classifying people by behaviour or traits leading to unjustified or disproportionate detrimental treatment.
        - Predictive policing that assesses the risk of someone committing a crime based solely on profiling or personality traits.
        - Untargeted scraping of facial images from the internet or CCTV to build or expand facial-recognition databases.
        - Emotion recognition in the workplace or education institutions (except for medical or safety reasons).
        - Biometric categorisation that infers race, political opinions, trade-union membership, religion, sex life or sexual orientation.
        - Real-time remote biometric identification in publicly accessible spaces for law enforcement (save narrow exceptions). Note: post/retrospective use is High, not prohibited.

        Keep the reason to one or two factual sentences. Set Category to the concrete basis, for example
        "Annex III(4) employment", "Article 5(1)(c) social scoring", "Article 50 transparency", or "Minimal risk".
        """;

    extension(ChatClientBuilder builder)
    {
        /// <summary>
        /// Adds the classification middleware to the pipeline so that every response is classified
        /// against the EU AI Act risk tiers.
        /// </summary>
        /// <returns>The <paramref name="builder"/> so that calls can be chained.</returns>
        public ChatClientBuilder UseEUAIActClassification() =>
            builder.Use(GetResponseAsync, GetStreamingResponseAsync);
    }

    extension(ChatResponse chatResponse)
    {
        /// <summary>
        /// Gets the EU AI Act classification attached to this response, or <see langword="null"/> if none was attached.
        /// </summary>
        /// <remarks>
        /// The classification can be attached either at the response level or at the last message level
        /// (the streaming path emits a trailing side-channel update; depending on whether it carries a
        /// <c>MessageId</c>, <c>ToChatResponse</c> routes the property onto the response or the last message).
        /// Both locations are checked.
        /// </remarks>
        public Classification? EUAIActClassification =>
            GetClassification(chatResponse.AdditionalProperties)
            ?? GetClassification(chatResponse.Messages.LastOrDefault()?.AdditionalProperties);
    }

    private static async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options,
        IChatClient inner,
        CancellationToken cancellationToken)
    {
        var messageList = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();

        var response = await inner.GetResponseAsync(messageList, options, cancellationToken);
        var classification = await ClassifySafelyAsync(inner, messageList.Concat(response.Messages), cancellationToken);

        var lastMessage = response.Messages.LastOrDefault();
        if (lastMessage is not null)
        {
            lastMessage.AdditionalProperties ??= [];
            lastMessage.AdditionalProperties[ClassificationKey] = classification;
        }

        response.AdditionalProperties ??= [];
        response.AdditionalProperties[ClassificationKey] = classification;

        return response;
    }

    private static async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options,
        IChatClient inner,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var messageList = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();

        // Forward every update untouched so the consumer's stream is never blocked by classification.
        List<ChatResponseUpdate> updates = [];
        await foreach (var update in inner.GetStreamingResponseAsync(messageList, options, cancellationToken))
        {
            updates.Add(update);
            yield return update;
        }

        // Once the stream has completed, classify the full conversation exactly once and emit the result as a
        // trailing side-channel update. Tagging it with the last message id makes ToChatResponse route the
        // classification onto that message; if there is none it falls back to the response level.
        var response = updates.ToChatResponse();
        var classification = await ClassifySafelyAsync(inner, messageList.Concat(response.Messages), cancellationToken);

        yield return new ChatResponseUpdate
        {
            MessageId = response.Messages.LastOrDefault()?.MessageId,
            AdditionalProperties = new() { [ClassificationKey] = classification },
        };
    }

    /// <summary>
    /// Runs the classification as a best-effort side channel. The primary chat response has already been
    /// produced by the time this runs, so any failure — including cancellation of the classification call —
    /// is recorded as <see cref="Risk.Unknown"/> rather than propagated, ensuring the primary response is
    /// never lost.
    /// </summary>
    private static async Task<Classification> ClassifySafelyAsync(
        IChatClient inner,
        IEnumerable<ChatMessage> conversation,
        CancellationToken cancellationToken)
    {
        try
        {
            return await ClassifyEUAIActRiskAsync(inner, conversation, cancellationToken);
        }
        catch (OperationCanceledException)
        {
            return Unknown("Classification was cancelled.");
        }
        catch (Exception ex)
        {
            return Unknown($"Classification failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Asks <paramref name="chatClient"/> to classify the EU AI Act risk tier of an entire conversation
    /// (the user's request together with the assistant's response).
    /// </summary>
    /// <param name="conversation">The conversation to classify.</param>
    /// <param name="cancellationToken">A token to cancel the classification call.</param>
    /// <returns>
    /// The <see cref="Classification"/> inferred for the conversation. Returns a <see cref="Risk.Unknown"/>
    /// classification when there is no text to classify or the model does not return a usable result.
    /// </returns>
    private static async Task<Classification> ClassifyEUAIActRiskAsync(
        IChatClient chatClient,
        IEnumerable<ChatMessage> conversation,
        CancellationToken cancellationToken = default)
    {
        if (conversation is null) throw new ArgumentNullException(nameof(conversation));

        var transcript = conversation
            .Where(m => !string.IsNullOrWhiteSpace(m.Text))
            .Select(m => new { Role = m.Role.Value, m.Text })
            .ToArray();

        if (transcript.Length == 0)
        {
            return Unknown("There was no text content to classify.");
        }

        var classificationResponse = await chatClient.GetResponseAsync<Classification>(
            messages:
            [
                new ChatMessage(ChatRole.System, SystemPrompt),
                    new ChatMessage(ChatRole.User, $"Conversation to classify:\n{JsonSerializer.Serialize(transcript)}"),
            ],
            options: new()
            {
                Temperature = 0.0F,
            },
            cancellationToken: cancellationToken);

        return classificationResponse.TryGetResult(out var result) && result is not null
            ? result
            : Unknown("The classifier did not return a usable result.");
    }

    /// <summary>Reads the classification stored under <see cref="ClassificationKey"/>, if any.</summary>
    private static Classification? GetClassification(AdditionalPropertiesDictionary? additionalProperties) =>
        additionalProperties?.TryGetValue(ClassificationKey, out var value) == true
            ? value as Classification
            : null;

    private static Classification Unknown(string reason) =>
        new() { Risk = Risk.Unknown, Category = "Unknown", Reason = reason };
}
