using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OpenAI;

namespace EUAIActClassifier;

public abstract class ChatClientTest
{
    public static readonly JsonSerializerOptions JsonReadable = new()
    {
        WriteIndented = true,
        Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() }
    };

    protected async Task<ChatResponse> GenerateChatResponse(IChatClient chatClient, IEnumerable<ChatMessage> messages)
    {
        var invocation = chatClient.GetStreamingResponseAsync(
            messages: messages,
            options: new()
            {
                Temperature = 0.0F,
                Tools = [],
            });

        List<ChatResponseUpdate> updates = await invocation.ToListAsync();
        ChatResponse chatResponse = updates.ToChatResponse();

        foreach (var message in chatResponse.Messages)
        {
            Console.WriteLine($"{JsonSerializer.Serialize(message, JsonReadable)}");
        }

        return chatResponse;
    }

    protected IChatClient GetOpenAIChatClient()
    {
        var host = CreateHostBuilder().Build();
        var chatClient = host.Services.GetRequiredKeyedService<IChatClient>("Microsoft.Extensions.AI.OpenAI");
        return chatClient;
    }

    static IHostBuilder CreateHostBuilder() => Host.CreateDefaultBuilder()
    .ConfigureAppConfiguration((ctx, config) =>
    {
        config.AddUserSecrets<ChatClientTest>();
    })
    .ConfigureServices((ctx, services) =>
    {
        var config = ctx.Configuration;

        services.AddKeyedSingleton<IChatClient>("Microsoft.Extensions.AI.OpenAI", (sp, key) =>
        {
            var openAiClient = new OpenAIClient(config["OPENAI_API_KEY"]).GetChatClient("gpt-4o-mini");

            var client = openAiClient
                .AsIChatClient()
                .AsBuilder()
                .Use(client => new ClassificationMiddleware(client, client))
                .Build();

            return client;
        });

        // services.AddKeyedSingleton<IChatClient>("AWSSDK.Extensions.Bedrock.MEAI", (sp, key) =>
        // {
        // 	var runtime = new AmazonBedrockRuntimeClient(
        // 		awsAccessKeyId: config["AWSBedrockAccessKeyId"]!,
        // 		awsSecretAccessKey: config["AWSBedrockSecretAccessKey"]!,
        // 		region: Amazon.RegionEndpoint.GetBySystemName(config["AWSBedrockRegion"]!));

        // 	var client = runtime
        // 		.AsIChatClient("eu.anthropic.claude-sonnet-4-20250514-v1:0")
        // 		.AsBuilder()
        // 		.UseFunctionInvocation()
        // 		.Build();

        // 	return client;
        // });
    });
}
