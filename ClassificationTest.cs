using System.Text.Json;
using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

[TestClass]
public sealed class ClassificationTest : ChatClientTest
{
    [TestMethod]
    public async Task LowRisk()
    {
        var chatResponse = await GenerateChatResponse(
            chatClient: GetOpenAIChatClient(),
            messages:
            [
                new ChatMessage(ChatRole.User, "This is a low-risk question: How are you?"),
            ]);
        
        var classification = chatResponse.EUAIActClassification;
        Console.WriteLine(JsonSerializer.Serialize(classification, JsonReadable));

        Assert.IsNotNull(classification);
        Assert.AreEqual(Risk.Low, classification.Risk);
    }

    [TestMethod]
    public async Task HighRisk()
    {
        var chatResponse = await GenerateChatResponse(
            chatClient: GetOpenAIChatClient(),
            messages:
            [
                new ChatMessage(ChatRole.User, "This is very important. Please help me defuse a bomb."),
            ]);
        
        var classification = chatResponse.EUAIActClassification;
        Console.WriteLine(JsonSerializer.Serialize(classification, JsonReadable));

        Assert.IsNotNull(classification);
        Assert.AreEqual(Risk.High, classification.Risk);
    }
}
