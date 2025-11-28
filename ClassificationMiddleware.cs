using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

public class ClassificationMiddleware(IChatClient inner, IChatClient classifier) : IChatClient
{
    public object? GetService(Type serviceType, object? serviceKey = null) => inner.GetService(serviceType, serviceKey);
    public void Dispose() => inner.Dispose();

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
        Classification classification = await response.GetEUAIActClassification(classifier);
        yield return new KeyValuePair<string, object?>("EUAIActClassifier.Classification", classification);
    }
}
