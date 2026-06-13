# EU AI Act Classifier

A drop-in middleware for `Microsoft.Extensions.AI.IChatClient` that classifies every chat
conversation against the [EU AI Act](https://artificialintelligenceact.eu/) risk tiers and attaches
the result to the response — so you can see, log, or gate on the risk level of what your AI is being
asked to do.

It plugs into the standard `IChatClient` pipeline, works with any provider (OpenAI, Azure OpenAI,
Bedrock, …), and supports both streaming and non-streaming calls.

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
var response = await client.GetResponseAsync(
    messages:
    [
        new ChatMessage(ChatRole.User, "This is a harmless question: How are you?"),
    ]);

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
