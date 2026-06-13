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

    /// <summary>Set by MSTest; used to surface per-test diagnostics.</summary>
    public TestContext TestContext { get; set; } = null!;

    // The host (and the chat client it roots) is expensive to build and safe to share across the live tests,
    // so it is created once for the whole test assembly.
    private static readonly Lazy<IHost> SharedHost = new(() => CreateHostBuilder().Build());

    protected async Task<ChatResponse> GenerateChatResponse(IChatClient chatClient, IEnumerable<ChatMessage> messages, bool useStreaming = true)
    {
        // The generated text itself is not asserted on (only the attached classification is), so cap output
        // to keep the live test calls fast and cheap.
        ChatOptions options = new() { Temperature = 0.0F, MaxOutputTokens = 256 };

        ChatResponse chatResponse;
        if (useStreaming)
        {
            List<ChatResponseUpdate> updates = await chatClient.GetStreamingResponseAsync(messages, options).ToListAsync();
            chatResponse = updates.ToChatResponse();
        }
        else
        {
            chatResponse = await chatClient.GetResponseAsync(messages, options);
        }

        foreach (var message in chatResponse.Messages)
        {
            TestContext.WriteLine($"{JsonSerializer.Serialize(message, JsonReadable)}");
        }

        return chatResponse;
    }

    protected IChatClient GetOpenAIChatClient()
    {
        var host = SharedHost.Value;

        // Live tests require a real key. Surface a clear, skippable result rather than an opaque auth failure.
        if (string.IsNullOrWhiteSpace(host.Services.GetRequiredService<IConfiguration>()["OPENAI_API_KEY"]))
        {
            Assert.Inconclusive("OPENAI_API_KEY is not configured (dotnet user-secrets); skipping live LLM tests.");
        }

        return host.Services.GetRequiredKeyedService<IChatClient>("Microsoft.Extensions.AI.OpenAI");
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
                .UseEUAIActClassification()
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
