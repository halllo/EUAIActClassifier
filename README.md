# EU AI Act Classifier

In this experimental project I plan to provide the [EU AI Act Compliance Checker](
https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/) as a middleware for `Microsoft.Extensions.AI.IChatClient` to classify responses automatically.

```csharp
// Use the classification middleware
var client = openAiClient
    .AsIChatClient()
    .AsBuilder()
    .Use(client => new ClassificationMiddleware(client, client))
    .Build();

// Generate a response
var response = await client.GetResponseAsync(
    messages:
    [
        new ChatMessage(ChatRole.User, "This is a harmless question: How are you?"),
    ]);

// Check the classification result
Assert.AreEqual(Risk.Low, response.EUAIActClassification.Risk);
```

🚧 Work in progress...
