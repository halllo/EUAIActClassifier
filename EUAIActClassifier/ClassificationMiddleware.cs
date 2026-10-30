using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

/// <summary>
/// Middleware for <see cref="IChatClient"/> that classifies every conversation against the EU AI Act risk
/// tiers and attaches the result so it can be read back via <c>EUAIActClassification</c>.
/// </summary>
/// <remarks>
/// Classification is a best-effort side channel: it never throws into the primary chat response. If the
/// classification call fails it is recorded as <see cref="Risk.Unknown"/> instead.
/// <para>
/// Only a complete turn is classified. A conversation left waiting on a tool approval or on a tool result is not
/// one — see <see cref="IsAwaitingToolApproval"/> and <see cref="IsAwaitingToolResult"/> — and is recorded as
/// <see cref="Risk.Unknown"/> without a classification call, leaving the verdict to the turn that resumes and finishes.
/// </para>
/// </remarks>
public static class ClassificationMiddleware
{
    /// <summary>The key under which the classification is stored in an <see cref="AdditionalPropertiesDictionary"/>.</summary>
    internal const string ClassificationKey = "EUAIActClassifier.Classification";

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
        /// Adds the classification middleware to the pipeline so that every response that completes a turn is
        /// classified against the EU AI Act risk tiers.
        /// </summary>
        /// <remarks>
        /// A response that ends the turn half way — on a tool call without a result, or on a tool approval nobody
        /// has answered — carries no verdict at all and costs no classification call. In a tool-calling loop that is
        /// every response but the last, so the loop is classified once, at the model call that answers.
        /// </remarks>
        /// <param name="options">
        /// Optional configuration — a custom system prompt, a conversation filter and/or a sampling temperature.
        /// When <see langword="null"/>, the built-in prompt is used, the whole conversation is classified and the
        /// classifier runs at temperature 0.
        /// </param>
        /// <returns>The <paramref name="builder"/> so that calls can be chained.</returns>
        public ChatClientBuilder UseEUAIActClassification(ClassificationOptions? options = null) =>
            builder.Use(
                (messages, chatOptions, inner, ct) => GetResponseAsync(messages, chatOptions, inner, options, ct),
                (messages, chatOptions, inner, ct) => GetStreamingResponseAsync(messages, chatOptions, inner, options, ct));
    }

    extension(IChatClient classifierClient)
    {
        /// <summary>
        /// Classifies a complete conversation against the EU AI Act risk tiers, using this client purely as the
        /// classification engine. This is the same engine behind <c>UseEUAIActClassification()</c>, exposed
        /// so the classifier does not have to sit in an <see cref="IChatClient"/> pipeline.
        /// </summary>
        /// <param name="conversation">
        /// The full conversation to classify — the user's request together with the final response messages.
        /// </param>
        /// <param name="options">
        /// Optional configuration — a custom system prompt, a conversation filter and/or a sampling temperature.
        /// When <see langword="null"/>, the built-in prompt is used, the whole conversation is classified and the
        /// classifier runs at temperature 0.
        /// </param>
        /// <param name="cancellationToken">A token to cancel the classification call.</param>
        /// <returns>
        /// The inferred <see cref="Classification"/>. This is a best-effort side channel: it never throws, returning
        /// a <see cref="Risk.Unknown"/> verdict on failure, when there is nothing to classify, or when the
        /// conversation is still waiting on a tool approval or a tool result — the last two without calling the model at all.
        /// </returns>
        /// <remarks>
        /// Call this <b>once</b> per logical turn, over the conversation that completes it. The
        /// <paramref name="classifierClient"/> can be any client (often a cheaper, separate model) and need not be
        /// the one that produced the conversation.
        /// </remarks>
        public Task<Classification> ClassifyEUAIActRiskAsync(
            IEnumerable<ChatMessage> conversation,
            ClassificationOptions? options = null,
            CancellationToken cancellationToken = default)
            => ClassifySafelyAsync(classifierClient, conversation, options, cancellationToken);
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

    extension(ChatResponseUpdate update)
    {
        /// <summary>
        /// Gets the EU AI Act classification carried by this streaming update, or <see langword="null"/> if none.
        /// </summary>
        /// <remarks>
        /// The trailing side-channel update emitted by the streaming pipeline carries the verdict; intermediate
        /// content updates return <see langword="null"/>. Useful for reacting to the verdict as the stream completes
        /// without first collecting every update into a <see cref="ChatResponse"/>.
        /// </remarks>
        public Classification? EUAIActClassification => GetClassification(update.AdditionalProperties);
    }

    extension(AdditionalPropertiesDictionary additionalProperties)
    {
        /// <summary>
        /// Gets the EU AI Act classification stored in these additional properties, or <see langword="null"/> if none.
        /// </summary>
        /// <remarks>
        /// This is the building block for agent-level middleware. An agent framework that preserves the
        /// <see cref="AdditionalPropertiesDictionary"/> when it wraps each chat update surfaces the very same
        /// dictionary on its own update type, so an agent middleware can read the verdict via
        /// <c>update.AdditionalProperties?.EUAIActClassification</c> — without referencing this package's storage
        /// details, and without this package taking a dependency on any agent framework.
        /// </remarks>
        public Classification? EUAIActClassification => GetClassification(additionalProperties);
    }

    private static async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options,
        IChatClient inner,
        ClassificationOptions? classificationOptions,
        CancellationToken cancellationToken)
    {
        var messageList = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();

        var response = await inner.GetResponseAsync(messageList, options, cancellationToken).ConfigureAwait(false);

        var conversation = messageList.Concat(response.Messages).ToList();
        if (IsIncompleteTurn(conversation))
        {
            return response;
        }

        var classification = await ClassifySafelyAsync(inner, conversation, classificationOptions, cancellationToken).ConfigureAwait(false);

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
        ClassificationOptions? classificationOptions,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var messageList = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();

        // Forward every update untouched so the consumer's stream is never blocked by classification.
        List<ChatResponseUpdate> updates = [];
        await foreach (var update in inner.GetStreamingResponseAsync(messageList, options, cancellationToken).ConfigureAwait(false))
        {
            updates.Add(update);
            yield return update;
        }

        // Once the stream has completed, classify the full conversation exactly once and emit the result as a
        // trailing side-channel update. Tagging it with the last message id makes ToChatResponse route the
        // classification onto that message; if there is none it falls back to the response level.
        var response = updates.ToChatResponse();

        // A response that leaves the turn half done emits nothing: a verdict there would stamp every step of a
        // tool-calling loop, and the turn is classified at the response that completes it.
        var conversation = messageList.Concat(response.Messages).ToList();
        if (IsIncompleteTurn(conversation))
        {
            yield break;
        }

        var classification = await ClassifySafelyAsync(inner, conversation, classificationOptions, cancellationToken).ConfigureAwait(false);

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
        ClassificationOptions? options,
        CancellationToken cancellationToken)
    {
        try
        {
            return await ClassifyCoreAsync(inner, conversation, options, cancellationToken).ConfigureAwait(false);
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
    /// <param name="chatClient">The chat client used as the classification engine.</param>
    /// <param name="conversation">The conversation to classify.</param>
    /// <param name="options">Optional custom system prompt, conversation filter and/or temperature; <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">A token to cancel the classification call.</param>
    /// <returns>
    /// The <see cref="Classification"/> inferred for the conversation. Returns a <see cref="Risk.Unknown"/>
    /// classification when the conversation awaits a tool approval or a tool result, when there is no text to classify, or when the
    /// model does not return a usable result.
    /// </returns>
    private static async Task<Classification> ClassifyCoreAsync(
        IChatClient chatClient,
        IEnumerable<ChatMessage> conversation,
        ClassificationOptions? options,
        CancellationToken cancellationToken = default)
    {
        if (conversation is null) throw new ArgumentNullException(nameof(conversation));

        var materialized = conversation as IReadOnlyList<ChatMessage> ?? conversation.ToList();

        // Checked before the filter so a ConversationFilter cannot hide the pending approval or call by trimming it away.
        if (IsAwaitingToolApproval(materialized))
        {
            return Unknown("The turn is awaiting a tool approval, so it is not complete.");
        }

        if (IsAwaitingToolResult(materialized))
        {
            return Unknown("The turn is awaiting a tool result, so it is not complete.");
        }

        var filtered = options?.ConversationFilter is { } filter ? filter(materialized) : materialized;

        var transcript = filtered
            .Where(m => !string.IsNullOrWhiteSpace(m.Text))
            .Select(m => new { Role = m.Role.Value, m.Text })
            .ToArray();

        if (transcript.Length == 0)
        {
            return Unknown("There was no text content to classify.");
        }

        var systemPrompt = string.IsNullOrWhiteSpace(options?.SystemPrompt) ? SystemPrompt : options!.SystemPrompt!;

        // NaN is the opt-out sentinel: leave Temperature unset so the model's own default applies.
        var temperature = options?.Temperature ?? ClassificationOptions.DefaultTemperature;

        var classificationResponse = await chatClient.GetResponseAsync<Classification>(
            messages:
            [
                new ChatMessage(ChatRole.System, systemPrompt),
                new ChatMessage(ChatRole.User, $"Conversation to classify:\n{JsonSerializer.Serialize(transcript)}"),
            ],
            options: new()
            {
                Temperature = float.IsNaN(temperature) ? null : temperature,
            },
            cancellationToken: cancellationToken).ConfigureAwait(false);

        return classificationResponse.TryGetResult(out var result) && result is not null
            ? result
            : Unknown("The classifier did not return a usable result.");
    }

    /// <summary>Reads the classification stored under <see cref="ClassificationKey"/>, if any.</summary>
    /// <remarks>
    /// Falls back to a value-type scan: the verdict is always stored as a <see cref="Classification"/>, so even if
    /// the storage key changes or a host re-homes the property onto a different carrier, the value is still found.
    /// </remarks>
    internal static Classification? GetClassification(AdditionalPropertiesDictionary? additionalProperties)
    {
        if (additionalProperties is null) return null;

        return additionalProperties.TryGetValue(ClassificationKey, out var value) && value is Classification classification
            ? classification
            : additionalProperties.Values.OfType<Classification>().FirstOrDefault();
    }

    /// <summary>
    /// Whether the conversation stops half way through its turn — at a tool approval nobody has answered or at a tool
    /// call without a result. Such a turn is not classified, and the middleware and agent attach no verdict to it.
    /// </summary>
    internal static bool IsIncompleteTurn(IReadOnlyList<ChatMessage> conversation) =>
        IsAwaitingToolApproval(conversation) || IsAwaitingToolResult(conversation);

    /// <summary>
    /// Whether the conversation stops at a tool approval nobody has answered — a
    /// <see cref="ToolApprovalRequestContent"/> whose <c>RequestId</c> no
    /// <see cref="ToolApprovalResponseContent"/> in the conversation matches.
    /// </summary>
    /// <remarks>
    /// Such a conversation is half a turn: the model asked to call a tool that needs a human decision, the tool
    /// has not run and no answer exists. There is a use case to name but no use of it yet, and the pending call
    /// contributes no text, so a verdict here would judge the request alone — and it would be paid for again on
    /// the run that resumes and actually completes the turn. Matching ids rather than inspecting the last message
    /// keeps that resume classifiable: it carries the answered request, and an answered request is not a pending
    /// one.
    /// </remarks>
    private static bool IsAwaitingToolApproval(IReadOnlyList<ChatMessage> conversation)
    {
        HashSet<string>? answered = null;
        foreach (var message in conversation)
        {
            foreach (var response in message.Contents.OfType<ToolApprovalResponseContent>())
            {
                (answered ??= new HashSet<string>(StringComparer.Ordinal)).Add(response.RequestId);
            }
        }

        foreach (var message in conversation)
        {
            foreach (var request in message.Contents.OfType<ToolApprovalRequestContent>())
            {
                if (answered is null || !answered.Contains(request.RequestId))
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Whether the conversation stops at a tool call nobody has answered — a <see cref="FunctionCallContent"/>
    /// whose <c>CallId</c> no <see cref="FunctionResultContent"/> in the conversation matches.
    /// </summary>
    /// <remarks>
    /// The same half a turn as an open approval, reached another way: the run hands a tool call to its caller to
    /// execute — a client-side tool the host runs itself — and ends there. The answer comes on the next run, once
    /// the caller sends the result back, and that run is the one to classify. Tool calls the run executed itself
    /// carry their result in the same conversation, so they do not count as pending.
    /// </remarks>
    private static bool IsAwaitingToolResult(IReadOnlyList<ChatMessage> conversation)
    {
        HashSet<string>? answered = null;
        foreach (var message in conversation)
        {
            foreach (var result in message.Contents.OfType<FunctionResultContent>())
            {
                (answered ??= new HashSet<string>(StringComparer.Ordinal)).Add(result.CallId);
            }
        }

        foreach (var message in conversation)
        {
            foreach (var call in message.Contents.OfType<FunctionCallContent>())
            {
                if (answered is null || !answered.Contains(call.CallId))
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static Classification Unknown(string reason) =>
        new() { Risk = Risk.Unknown, Category = "Unknown", Reason = reason };
}
