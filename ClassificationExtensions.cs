using System.Text.Json;
using Microsoft.Extensions.AI;

public static class ClassificationExtensions
{
    extension(ChatResponse chatResponse)
    {
        public Classification? EUAIActClassification => chatResponse.AdditionalProperties.EUAIActClassification;
    }

    extension(AdditionalPropertiesDictionary? additionalProperties)
    {
        public Classification? EUAIActClassification => additionalProperties?["EUAIActClassifier.Classification"] as Classification;
    }
}