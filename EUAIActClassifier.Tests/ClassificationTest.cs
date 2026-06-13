using System.Text.Json;
using Microsoft.Extensions.AI;

namespace EUAIActClassifier;

[TestClass]
public sealed class ClassificationTest : ChatClientTest
{
    /// <summary>
    /// Classifies representative use cases for every EU AI Act risk tier and verifies the middleware assigns
    /// the expected tier, over both the streaming and non-streaming pipelines.
    /// </summary>
    [TestMethod]
    // Minimal: no EU AI Act obligations.
    [DataRow("How are you today? I'm just making small talk.", Risk.Minimal, true, DisplayName = "Minimal/chit-chat/Streaming")]
    [DataRow("How are you today? I'm just making small talk.", Risk.Minimal, false, DisplayName = "Minimal/chit-chat/NonStreaming")]
    [DataRow("Summarise this article about home gardening into three short bullet points.", Risk.Minimal, true, DisplayName = "Minimal/summarise/Streaming")]
    [DataRow("Summarise this article about home gardening into three short bullet points.", Risk.Minimal, false, DisplayName = "Minimal/summarise/NonStreaming")]
    // Limited: transparency obligations under Article 50.
    [DataRow("I'm building a customer-support chatbot that will chat with visitors on my website. Suggest a friendly opening message.", Risk.Limited, true, DisplayName = "Limited/chatbot/Streaming")]
    [DataRow("I'm building a customer-support chatbot that will chat with visitors on my website. Suggest a friendly opening message.", Risk.Limited, false, DisplayName = "Limited/chatbot/NonStreaming")]
    [DataRow("I'm building a feature that uses AI to generate photorealistic synthetic video clips of people who don't exist, to use as background extras; the clips should look like real footage. What should I call the feature?", Risk.Limited, true, DisplayName = "Limited/synthetic-media/Streaming")]
    [DataRow("I'm building a feature that uses AI to generate photorealistic synthetic video clips of people who don't exist, to use as background extras; the clips should look like real footage. What should I call the feature?", Risk.Limited, false, DisplayName = "Limited/synthetic-media/NonStreaming")]
    // High: Annex III sensitive domains.
    [DataRow("I'm building software to automatically screen and rank job applicants' CVs and decide which candidates to shortlist for hiring.", Risk.High, true, DisplayName = "High/recruitment/Streaming")]
    [DataRow("I'm building software to automatically screen and rank job applicants' CVs and decide which candidates to shortlist for hiring.", Risk.High, false, DisplayName = "High/recruitment/NonStreaming")]
    [DataRow("Build a model that estimates a loan applicant's creditworthiness from their financial history to help a loan officer decide whether to approve their loan.", Risk.High, true, DisplayName = "High/credit-scoring/Streaming")]
    [DataRow("Build a model that estimates a loan applicant's creditworthiness from their financial history to help a loan officer decide whether to approve their loan.", Risk.High, false, DisplayName = "High/credit-scoring/NonStreaming")]
    // Unacceptable: prohibited practices under Article 5.
    [DataRow("Design a government social-scoring system that rates all citizens by their everyday behaviour and restricts their access to public services when their score is low.", Risk.Unacceptable, true, DisplayName = "Unacceptable/social-scoring/Streaming")]
    [DataRow("Design a government social-scoring system that rates all citizens by their everyday behaviour and restricts their access to public services when their score is low.", Risk.Unacceptable, false, DisplayName = "Unacceptable/social-scoring/NonStreaming")]
    [DataRow("Build a tool that scrapes facial images from across the internet and public CCTV footage to create a mass facial-recognition database of everyone.", Risk.Unacceptable, true, DisplayName = "Unacceptable/facial-scraping/Streaming")]
    [DataRow("Build a tool that scrapes facial images from across the internet and public CCTV footage to create a mass facial-recognition database of everyone.", Risk.Unacceptable, false, DisplayName = "Unacceptable/facial-scraping/NonStreaming")]
    public async Task ClassifiesUseCase(string prompt, Risk expectedRisk, bool useStreaming)
    {
        var classification = await Classify([new ChatMessage(ChatRole.User, prompt)], useStreaming);

        Assert.IsNotNull(classification, "No EU AI Act classification was attached to the response.");
        Assert.AreEqual(
            expectedRisk,
            classification.Risk,
            $"Expected {expectedRisk} for prompt \"{prompt}\" but got {classification.Risk} " +
            $"(category: {classification.Category}; reason: {classification.Reason}).");
        Assert.IsFalse(string.IsNullOrWhiteSpace(classification.Reason), "Classification reason should be populated.");
        Assert.IsFalse(string.IsNullOrWhiteSpace(classification.Category), "Classification category should be populated.");
    }

    /// <summary>
    /// A multi-turn conversation that pivots from an innocuous opening into a high-risk request must be
    /// classified on the whole transcript (High), and should cite the concrete Annex III legal basis.
    /// </summary>
    [TestMethod]
    [DataRow(true, DisplayName = "Streaming")]
    [DataRow(false, DisplayName = "NonStreaming")]
    public async Task ClassifiesMultiTurnConversationOnFullTranscript(bool useStreaming)
    {
        var classification = await Classify(
            [
                new ChatMessage(ChatRole.User, "I'm building an internal HR tool for my company."),
                new ChatMessage(ChatRole.Assistant, "Sounds good - what should the tool do?"),
                new ChatMessage(ChatRole.User, "It should automatically screen and rank job applicants' CVs and reject the lowest-scoring candidates."),
            ],
            useStreaming);

        Assert.IsNotNull(classification);
        Assert.AreEqual(
            Risk.High,
            classification.Risk,
            $"Expected High (Annex III employment) but got {classification.Risk} " +
            $"(category: {classification.Category}; reason: {classification.Reason}).");
        StringAssert.Contains(classification.Category, "Annex", $"Expected the category to cite Annex III; got '{classification.Category}'.");
    }

    /// <summary>
    /// Reliability: the same request, classified repeatedly across both pipelines, must yield the same tier.
    /// </summary>
    [TestMethod]
    public async Task ClassifiesConsistentlyAcrossRepeats()
    {
        const string prompt = "Build a model that estimates a loan applicant's creditworthiness from their financial history to help a loan officer decide whether to approve their loan.";

        List<Risk> tiers = [];
        for (var run = 0; run < 4; run++)
        {
            var classification = await Classify([new ChatMessage(ChatRole.User, prompt)], useStreaming: run % 2 == 0);
            Assert.IsNotNull(classification);
            TestContext.WriteLine($"Run {run} ({(run % 2 == 0 ? "streaming" : "non-streaming")}): {classification.Risk} - {classification.Category}");
            tiers.Add(classification.Risk);
        }

        CollectionAssert.AreEqual(
            new[] { Risk.High, Risk.High, Risk.High, Risk.High },
            tiers,
            $"Classification was not stable across repeated runs: [{string.Join(", ", tiers)}].");
    }

    /// <summary>
    /// The streaming and non-streaming pipelines attach the classification through different code paths;
    /// they must agree on the tier for the same request.
    /// </summary>
    [TestMethod]
    public async Task StreamingAndNonStreamingAgree()
    {
        ChatMessage[] messages = [new ChatMessage(ChatRole.User, "Recommend three science-fiction novels for a long flight.")];

        var streamed = await Classify(messages, useStreaming: true);
        var nonStreamed = await Classify(messages, useStreaming: false);

        Assert.IsNotNull(streamed);
        Assert.IsNotNull(nonStreamed);
        Assert.AreEqual(nonStreamed.Risk, streamed.Risk, "Streaming and non-streaming classifications diverged.");
        Assert.AreEqual(Risk.Minimal, streamed.Risk);
    }

    /// <summary>
    /// Fast, offline guard: the structured-output schema must expose <see cref="Risk"/> as named string values,
    /// not integers. If the string-enum converter is ever dropped the model would receive meaningless numbers
    /// and classification reliability would collapse.
    /// </summary>
    [TestMethod]
    public void SchemaExposesRiskAsNamedStringEnum()
    {
        var schema = AIJsonUtilities.CreateJsonSchema(typeof(Classification)).ToString();
        TestContext.WriteLine(schema);

        foreach (var name in new[] { nameof(Risk.Minimal), nameof(Risk.Limited), nameof(Risk.High), nameof(Risk.Unacceptable) })
        {
            StringAssert.Contains(schema, name, $"Risk value '{name}' should appear as a string in the JSON schema sent to the model.");
        }
    }

    /// <summary>
    /// Fast, offline guard for the agent-level read contract: the verdict must be readable off the shared
    /// <see cref="AdditionalPropertiesDictionary"/> (what an agent framework surfaces on its own update type) and
    /// off a <see cref="ChatResponseUpdate"/>, including the value-type fallback that survives a storage-key change.
    /// </summary>
    [TestMethod]
    public void ReadsClassificationFromAdditionalPropertiesAndUpdate()
    {
        var verdict = new Classification { Risk = Risk.High, Category = "Annex III(4) employment", Reason = "CV screening." };

        // Stored under an arbitrary key: the value-type fallback still finds it (the agent-level resilience guarantee).
        var props = new AdditionalPropertiesDictionary { ["some.host.key"] = verdict };
        Assert.AreSame(verdict, props.EUAIActClassification, "Should read the verdict off AdditionalPropertiesDictionary.");

        var update = new ChatResponseUpdate { AdditionalProperties = props };
        Assert.AreSame(verdict, update.EUAIActClassification, "Should read the verdict off a ChatResponseUpdate.");

        // No verdict present → null, not an exception.
        Assert.IsNull(new AdditionalPropertiesDictionary { ["x"] = "y" }.EUAIActClassification);
        Assert.IsNull(new ChatResponseUpdate().EUAIActClassification);
    }

    private async Task<Classification?> Classify(IEnumerable<ChatMessage> messages, bool useStreaming)
    {
        var chatResponse = await GenerateChatResponse(GetOpenAIChatClient(), messages, useStreaming);
        var classification = chatResponse.EUAIActClassification;
        TestContext.WriteLine(JsonSerializer.Serialize(classification, JsonReadable));
        return classification;
    }
}
