using ModelDesk.Infrastructure.Runtime;

namespace ModelDesk.Tests;

public sealed class FlashProfileTests
{
    [Fact]
    public void NamedFlashProfileUsesStableKeyLabelAndIndependentPaths()
    {
        var root = Path.Combine(Path.GetTempPath(), "ModelDesk flash profile", Guid.NewGuid().ToString("N"));
        try
        {
            var directory = Path.Combine(root, "config", "models");
            Directory.CreateDirectory(directory);
            Directory.CreateDirectory(Path.Combine(root, "glm_local"));
            File.WriteAllText(Path.Combine(root, "glm_local", "__main__.py"), "");
            File.WriteAllText(Path.Combine(directory, "flash-cybersecurity-w4a16.json"), """
                {
                  "model_id": "dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16",
                  "revision": "4c2ee763ff75f3ba50652ee0f024b0b630cd54a0",
                  "model_directory": "models/GLM-5.3-Flash-CYBERSECURITY-W4A16",
                  "report_namespace": "flash-w4a16",
                  "inference_backend": "openai",
                  "backend_status": "external_openai_required"
                }
                """);
            var profile = Assert.Single(new PythonCoreService().GetProfiles(root));
            Assert.Equal("flash-w4a16", profile.Key);
            Assert.Equal("GLM Flash · W4A16", profile.DisplayName);
            Assert.Equal("dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16", profile.ModelId);
            Assert.Equal("4c2ee763ff75f3ba50652ee0f024b0b630cd54a0", profile.Revision);
            Assert.Equal(Path.Combine(root, "models", "GLM-5.3-Flash-CYBERSECURITY-W4A16"), profile.ModelDirectory);
            Assert.Equal(Path.Combine(root, "reports", "flash-w4a16"), profile.ReportDirectory);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }
}
