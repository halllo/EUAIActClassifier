using System.Text.Json;
using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

[TestClass]
public sealed class ClassifierTest : ChatClientTest
{
    [TestMethod]
    public async Task Unclassified()
    {
        var chatResponse = GenerateChatResponse(
            chatClient: GetOpenAIChatClient(),
            messages:
            [
                new ChatMessage(ChatRole.User, "This is a low-risk question: How are you?"),
            ]);
        
        Assert.IsTrue(chatResponse.AdditionalProperties?.ContainsKey("EUAIActClassifier.Class"));
    }
}
