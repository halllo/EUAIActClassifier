# EU AI Act Classifier

In this experimental project I plan to provide the [EU AI Act Compliance Checker](
https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/) as a middleware for `Microsoft.Extensions.AI.IChatClient` to classify responses automatically.

```csharp
var response = await client.GetResponseAsync(
    messages:
    [
        new ChatMessage(ChatRole.User, "This is a harmless question: How are you?"),
    ]);

Assert.AreEqual(Risk.Low, response.EUAIActClassification.Risk);
```

🚧 Work in progress...
