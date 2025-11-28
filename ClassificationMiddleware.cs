using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;

public partial class ClassificationMiddleware(IChatClient inner, IChatClient classifier) : IChatClient
{
    public void Dispose()
    {
        inner.Dispose();
    }

    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var response = await inner.GetResponseAsync(messages, options, cancellationToken);
        await foreach (var kvp in GetClassification(response))
        {
            var lastMessage = response.Messages.Last();
            lastMessage.AdditionalProperties ??= [];
            lastMessage.AdditionalProperties[kvp.Key] = kvp.Value;

            response.AdditionalProperties ??= [];
            response.AdditionalProperties[kvp.Key] = kvp.Value;
        }
        return response;
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        return inner.GetService(serviceType, serviceKey);
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        List<ChatResponseUpdate> updates = [];
        List<AIContent> contents = [];
        await foreach (var update in inner.GetStreamingResponseAsync(messages, options, cancellationToken))
        {
            updates.Add(update);
            foreach (var content in update.Contents)
            {
                contents.Add(content);

                if (content is UsageContent usage) //detecting last event
                {
                    var response = updates.ToChatResponse();
                    await foreach (var kvp in GetClassification(response))
                    {
                        update.AdditionalProperties ??= [];
                        update.AdditionalProperties[kvp.Key] = kvp.Value;
                    }
                }
            }
            yield return update;
        }
    }

    private async IAsyncEnumerable<KeyValuePair<string, object?>> GetClassification(ChatResponse response)
    {
        Classification classification = await Classify(response);
        yield return new KeyValuePair<string, object?>("EUAIActClassifier.Classification", classification);
    }

    private async Task<Classification> Classify(ChatResponse response)
    {
        // TODO: classify based on https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/
        var classificationResponse = await classifier.GetResponseAsync<Classification>(
            messages:
            [
                new ChatMessage(ChatRole.System, "Classify the users request. Respond as concise and succinct as possible. The shorter the better."),
                new ChatMessage(ChatRole.User, $"Request: {JsonSerializer.Serialize(response.Messages.Select(m => new { m.Role, m.Text }))}"),
            ],
            options: new()
            {
                Temperature = 0.0F,
            }
        );
        return classificationResponse.Result;
    }
}
