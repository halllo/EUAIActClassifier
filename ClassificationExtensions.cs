using System.Text.Json;
using Microsoft.Extensions.AI;

public static class ClassificationExtensions
{
    extension(ChatResponse chatResponse)
    {
        public async Task<Classification> GetEUAIActClassification(IChatClient chatClient)
        {
            // TODO: classify based on https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/
            var classificationResponse = await chatClient.GetResponseAsync<Classification>(
                messages:
                [
                    new ChatMessage(ChatRole.System, "Classify the users request. Respond as concise and succinct as possible. The shorter the better."),
                    new ChatMessage(ChatRole.User, $"Request: {JsonSerializer.Serialize(chatResponse.Messages.Select(m => new { m.Role, m.Text }))}"),
                ],
                options: new()
                {
                    Temperature = 0.0F,
                }
            );
            return classificationResponse.Result;
        }

        public Classification? EUAIActClassification => chatResponse.AdditionalProperties.EUAIActClassification;
    }

    extension(AdditionalPropertiesDictionary? additionalProperties)
    {
        public Classification? EUAIActClassification => additionalProperties?["EUAIActClassifier.Classification"] as Classification;
    }
}