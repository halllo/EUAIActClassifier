# EU AI Act Classifier

A drop-in middleware that classifies every chat conversation against the
[EU AI Act](https://artificialintelligenceact.eu/) risk tiers and attaches the result to the response —
so you can see, log, or gate on the risk level of what your AI is being asked to do.

It works at two levels: as `Microsoft.Extensions.AI.IChatClient` middleware (any provider — OpenAI,
Azure OpenAI, Bedrock, …), and as a first-class **Microsoft Agent Framework** agent that classifies once
per agent run. Both support streaming and non-streaming calls.

## Installation

The package is published on [NuGet](https://www.nuget.org/packages/EUAIActClassifier).

```bash
dotnet add package EUAIActClassifier
```

The package targets `netstandard2.0`, so it works with .NET Framework 4.6.1+, .NET Core, and modern
.NET. It depends on [`Microsoft.Extensions.AI`](https://www.nuget.org/packages/Microsoft.Extensions.AI)
and the lightweight [`Microsoft.Agents.AI.Abstractions`](https://www.nuget.org/packages/Microsoft.Agents.AI.Abstractions)
(for the agent-level middleware) — **not** the full agent runtime — both pulled in automatically.

## How it works

The middleware passes your request straight through to the underlying model, then makes a second,
structured-output call that classifies the **whole conversation** (your request *and* the model's
reply) into one risk tier. The verdict is attached to the response and read back via
`response.EUAIActClassification`.

```text
   ┌──────────┐   request    ┌────────────────────────────┐   request    ┌──────────────┐
   │  Caller  │ ───────────▶ │  UseEUAIActClassification  │ ───────────▶ │ inner client │
   │          │ ◀─────────── │        (middleware)        │ ◀─────────── │    (LLM)     │
   └──────────┘  response +  └────────────────────────────┘   response   └──────────────┘
                Classification          │       ▲
                                        │       │  Classification { Risk, Category, Reason }
                                        ▼       │
                              ┌────────────────────────────┐
                              │  classify(request + reply) │   second LLM call, temperature 0,
                              │  → structured output       │   JSON-schema-constrained
                              └────────────────────────────┘
```

Classification is a **best-effort side channel**: it never throws into your primary response. If the
classifier call fails (or there is nothing to classify), the result is recorded as `Risk.Unknown`
rather than surfacing an error.

The [Microsoft Agent Framework](#microsoft-agent-framework-first-class) variant works the same way but
runs at the **agent** level — it classifies once over the *completed* conversation of an agent run (after
any tool-calling round-trips), rather than per model call. See that section for why.

## Risk tiers

The EU AI Act takes a risk-based approach — the higher the tier, the heavier the obligations, and the
fewer systems fall into it:

```text
              /\
             /  \          Unacceptable  —  prohibited outright       (Art. 5)
            /----\
           /      \        High          —  strict obligations  (Art. 6 / Annex III)
          /--------\
         /          \      Limited       —  transparency duties       (Art. 50)
        /------------\
       /              \    Minimal       —  no obligations (the default)
      /----------------\
```

Every conversation is mapped to one of these tiers (see [`Risk`](EUAIActClassifier/Risk.cs)):

|Tier|Legal basis|Examples|
|----|-----------|--------|
|`Minimal`|— (default)|general Q&A, summarisation, coding help, spam filters, AI in games|
|`Limited`|[Article 50](https://artificialintelligenceact.eu/article/50/)|chatbots, deepfakes / synthetic media (must disclose)|
|`High`|[Article 6 & Annex III](https://artificialintelligenceact.eu/annex/3/)|remote biometrics, critical infrastructure, education grading, hiring, credit scoring, law enforcement, migration, justice|
|`Unacceptable`|[Article 5](https://artificialintelligenceact.eu/article/5/)|social scoring, manipulation, untargeted facial-image scraping, predictive policing by profiling|

## Usage

```csharp
// Add the classification middleware to any IChatClient pipeline.
var client = openAiClient
    .AsIChatClient()
    .AsBuilder()
    .UseEUAIActClassification()
    .Build();

// Use the client exactly as before.
ChatMessage[] messages = [new ChatMessage(ChatRole.User, "This is a harmless question: How are you?")];
var response = await client.GetResponseAsync(messages);

// Read the classification attached to the response.
var classification = response.EUAIActClassification;
Console.WriteLine(classification?.Risk);      // Minimal
Console.WriteLine(classification?.Category);  // e.g. "Minimal risk – general assistance"
Console.WriteLine(classification?.Reason);    // short justification
```

The same getter works for streaming calls — collect the updates into a response and read it back:

```csharp
var updates = await client.GetStreamingResponseAsync(messages).ToListAsync();
var risk = updates.ToChatResponse().EUAIActClassification?.Risk;
```

If you want to **react** to the verdict as the stream completes (log it, gate on it, surface it in a UI)
without first aggregating the whole turn, read it off the individual update with `update.EUAIActClassification`:

```csharp
await foreach (var update in client.GetStreamingResponseAsync(messages))
{
    // ... forward the model's content to your UI as it arrives ...

    if (update.EUAIActClassification is { Risk: >= Risk.High } verdict)
    {
        // React: the turn was classified High risk or above.
        logger.LogWarning("EU AI Act risk {Risk}: {Reason}", verdict.Risk, verdict.Reason);
    }
}
```

**Streaming contract:**

- The verdict rides on a **trailing side-channel update** emitted *after* the model's own output has
  streamed; intermediate content updates return `null` from `EUAIActClassification`.
- Classification is **best-effort**: on classifier failure the verdict is present but `Risk` is
  `Risk.Unknown` (it never throws into your stream), so handle `Unknown` explicitly.

## Microsoft Agent Framework (first-class)

For [Microsoft Agent Framework](https://www.nuget.org/packages/Microsoft.Agents.AI) agents, classify at the
**agent** level — not in the agent's `IChatClient` pipeline.

> **Why not the `IChatClient` pipeline?** An agent's run loop (e.g. `ChatClientAgent` +
> `FunctionInvokingChatClient`) calls the underlying `IChatClient` **once per model round-trip**, looping
> `model → tool → model → …` until it is done. `IChatClient` middleware sits *inside* that loop, so it would
> classify **every intermediate step** (each one an extra LLM call) instead of the finished turn. Microsoft's
> docs are explicit: chat-client middleware *"executes for each model call, including calls that send tool
> results back to the model during a multi-turn tool calling sequence."*

`UseEUAIActClassification(classifier)` wraps any `AIAgent` so each run is classified **exactly once**, over
the completed conversation:

```csharp
// `classifier` is any IChatClient used purely as the classification engine — it is NOT in the agent's
// inference path (often a cheaper, separate model). Pass the base agent through the wrapper:
AIAgent agent = baseAgent.UseEUAIActClassification(classifier);

// Or compose it into a builder pipeline:
AIAgent agent = baseAgent
    .AsBuilder()
    .Use(inner => inner.UseEUAIActClassification(classifier))
    .Build();

// Run as usual; read the verdict off the response or the streamed updates with the agent-side getters.
var response = await agent.RunAsync("Screen these CVs and shortlist the best.");
var risk = response.EUAIActClassification?.Risk;   // High

await foreach (var update in agent.RunStreamingAsync("Screen these CVs and shortlist the best."))
{
    if (update.EUAIActClassification is { Risk: >= Risk.High } verdict)
    {
        // React: a trailing side-channel update carries the verdict once the run completes.
    }
}
```

`AgentResponse.EUAIActClassification` and `AgentResponseUpdate.EUAIActClassification` mirror the getters on the
`IChatClient` side. This is built on the lightweight `Microsoft.Agents.AI.Abstractions` (the `DelegatingAIAgent`
decorator base), so it works with **any** `AIAgent` and pulls in only the abstractions — not the full agent runtime.

If you'd rather drive the engine yourself, it is also exposed directly as
`classifier.ClassifyEUAIActRiskAsync(conversation)` — give it any `IChatClient` and the full conversation, and
it returns the verdict as the same best-effort side channel (never throws; `Risk.Unknown` on failure).

## The result

[`Classification`](EUAIActClassifier/Classification.cs) carries three fields:

|Field|Type|Meaning|
|-----|----|-------|
|`Risk`|[`Risk`](EUAIActClassifier/Risk.cs)|the risk tier (`Minimal` / `Limited` / `High` / `Unacceptable`, or `Unknown` on failure)|
|`Category`|`string`|the concrete legal basis, e.g. `"Annex III(4) employment"` or `"Article 5(1)(c) social scoring"`|
|`Reason`|`string`|a one- or two-sentence justification for the tier|

## Notes

- **Classifies the use case, not the topic.** A request that merely *mentions* a sensitive subject
  stays `Minimal` if the AI is only acting as a general assistant; the tier reflects the AI system's
  purpose, as the Act intends.
- **Requires a structured-output-capable model** for the classification call (it is constrained to a
  JSON schema and run at `temperature = 0` for stability).
- General-purpose AI model obligations (Chapter V) are out of scope — the middleware judges the
  **use-case** risk tier of a conversation, not model-provider duties.
- This is a classification aid, not legal advice or a certified compliance assessment.
