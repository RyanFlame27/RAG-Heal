"""
LLM wrapper for 4-bit quantized Mistral 7B (or Llama 3.1 8B).

Loads the model using bitsandbytes for memory-efficient inference
on Colab free-tier GPUs. Supports returning token-level logits
for the entropy-based confidence detector.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


class LLMWrapper:
    """
    Wrapper around a 4-bit quantized causal LM.

    Provides a simple `generate()` interface that optionally returns
    per-token log probabilities for downstream confidence detection.
    """

    def __init__(
        self,
        model_name: str = config.LLM_MODEL_NAME,
        quantization_bits: int = config.LLM_QUANTIZATION_BITS,
        device_map: str = "auto",
    ):
        """
        Initialize and load the quantized LLM.

        Args:
            model_name: HuggingFace model ID.
            quantization_bits: Number of bits for quantization (4 or 8).
            device_map: Device mapping strategy.
        """
        self.model_name = model_name
        print(f"Loading LLM: {model_name} ({quantization_bits}-bit quantized)...")

        # Configure quantization
        if quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization_bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model on explicit GPU device if available
        if device_map == "auto" and torch.cuda.is_available():
            target_device_map = "cuda:0"
        else:
            target_device_map = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=target_device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print(f"LLM loaded successfully on device: {self.model.device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = config.LLM_MAX_NEW_TOKENS,
        temperature: float = config.LLM_TEMPERATURE,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.

        Args:
            prompt: The full prompt string (including context).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (low = deterministic).
            return_logits: If True, return per-token log probabilities.

        Returns:
            Dict with keys:
                - text: Generated text string.
                - token_logits: (optional) List of token-level log prob dicts.
                - input_length: Number of input tokens (for offset calculations).
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "output_logits": return_logits,
            "return_dict_in_generate": return_logits,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            if return_logits:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                generated_ids = outputs.sequences[0][input_length:]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                # Extract per-token log probabilities
                # Raw logits keep uncertainty comparable across sampling temperatures.
                token_logits = self._extract_token_logits(outputs.logits)
            else:
                output_ids = self.model.generate(**inputs, **gen_kwargs)
                generated_ids = output_ids[0][input_length:]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                token_logits = None

        result = {
            "text": generated_text.strip(),
            "input_length": input_length,
        }

        if return_logits and token_logits is not None:
            result["token_logits"] = token_logits

        return result

    def _extract_token_logits(
        self,
        scores: tuple,
    ) -> List[Dict[str, Any]]:
        """
        Extract per-token entropy and top-k probabilities from generation scores.

        Args:
            scores: Tuple of score tensors from model.generate()

        Returns:
            List of dicts with entropy, top_prob, and token info per generated token.
        """
        token_info = []

        for step_scores in scores:
            # step_scores shape: (batch_size, vocab_size)
            logits = step_scores[0].float()
            if (torch.isnan(logits).any() or torch.isposinf(logits).any()
                    or not torch.isfinite(logits).any()):
                token_info.append({"entropy": None, "top_prob": None})
                continue
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Entropy in nats; omit zero-mass terms to avoid 0 * -inf = NaN.
            positive = probs > 0
            entropy = -(probs[positive] * log_probs[positive]).sum().item()

            # Top token probability
            top_prob, top_idx = probs.max(dim=-1)

            token_info.append({
                "entropy": entropy,
                "top_prob": top_prob.item(),
                "top_token_id": top_idx.item(),
                "top_token": self.tokenizer.decode([top_idx.item()]),
            })

        return token_info

    def build_rag_prompt(
        self,
        query: str,
        context_chunks: List[str],
    ) -> str:
        """
        Build a RAG prompt from query and retrieved context chunks.

        Args:
            query: User's question.
            context_chunks: List of retrieved text chunks.

        Returns:
            Formatted prompt string.
        """
        context = "\n\n---\n\n".join(context_chunks)
        return config.RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )


if __name__ == "__main__":
    # This will only work on a machine with GPU + enough VRAM
    print("Testing LLM wrapper...")
    try:
        llm = LLMWrapper()
        prompt = llm.build_rag_prompt(
            query="What is the capital of France?",
            context_chunks=["Paris is the capital and most populous city of France."],
        )
        result = llm.generate(prompt, return_logits=True)
        print(f"\nGenerated: {result['text']}")
        if result.get("token_logits"):
            avg_entropy = np.mean([t["entropy"] for t in result["token_logits"]])
            print(f"Average token entropy: {avg_entropy:.4f}")
    except Exception as e:
        print(f"LLM test failed (expected if no GPU): {e}")
