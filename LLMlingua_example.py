# LLMlingua let you simplify prompt to use fewer tokens for cost optimization.

from llmlingua import PromptCompressor

compressor = PromptCompressor(model_name="longllmlingua")

compressed_context = compressor.compress_prompt(
    context,
    rate=0.2
)
