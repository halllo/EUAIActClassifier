using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

public class Classifier(IChatClient inner) : IChatClient
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
            response.AdditionalProperties ??= new AdditionalPropertiesDictionary();
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
        await foreach (var update in inner.GetStreamingResponseAsync(messages, options, cancellationToken))
        {
            await foreach (var kvp in GetClassification(update))
            {
                update.AdditionalProperties ??= new AdditionalPropertiesDictionary();
                update.AdditionalProperties[kvp.Key] = kvp.Value;
            }
            yield return update;
        }
    }

    private static async IAsyncEnumerable<KeyValuePair<string, object?>> GetClassification(ChatResponse response)
    {
        await Task.CompletedTask;
        yield return new KeyValuePair<string, object?>("EUAIActClassifier.Class", "unclassified");
    }

    private static async IAsyncEnumerable<KeyValuePair<string, object?>> GetClassification(ChatResponseUpdate update)
    {
        await Task.CompletedTask;
        yield return new KeyValuePair<string, object?>("EUAIActClassifier.Class", "unclassified");
    }
}