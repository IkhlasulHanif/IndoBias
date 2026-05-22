"""Hugging Face chat model implementation with Apple Silicon support."""

import inspect
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFChatModel:
    """Generic HF chat model with MPS (Metal Performance Shaders) support."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: Optional[str] = None,
        max_length: int = 1024,
        num_generations: int = 1,
        temperature: float = 0.0,
        top_p: float = 0.9,
        enable_thinking: bool = False,
    ):
        """
        Initialize an HF chat model with MPS support.

        Args:
            model_name: HuggingFace model name/path
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto-detect)
            max_length: Maximum generation length
            num_generations: Default number of responses to sample per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            enable_thinking: Whether to allow models that support reasoning mode
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_generations = num_generations
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(
            f"Loading {model_name} on {self.device} "
            f"(enable_thinking={self.enable_thinking})..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self._supports_enable_thinking():
            self.chat_template_kwargs["enable_thinking"] = enable_thinking
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device != "cpu" else torch.float32

        if self.device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=self.device if self.device == "cuda" else "auto",
                trust_remote_code=True,
            )

        self.model.eval()
        print(f"✓ Model loaded successfully on {self.device}")

    def _supports_enable_thinking(self) -> bool:
        """Detect whether the tokenizer chat template accepts enable_thinking."""
        try:
            signature = inspect.signature(self.tokenizer.apply_chat_template)
        except (TypeError, ValueError):
            return False
        return "enable_thinking" in signature.parameters

    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            **self.chat_template_kwargs,
        )

    def _build_generation_kwargs(
        self,
        max_new_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        num_generations = num_generations or self.num_generations
        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_top_p = self.top_p if top_p is None else top_p
        do_sample = resolved_temperature > 0

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_length,
            "num_return_sequences": num_generations,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = resolved_temperature
            gen_kwargs["top_p"] = resolved_top_p

        gen_kwargs.update(kwargs)
        return gen_kwargs

    def _move_inputs_to_device(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _strip_prompt_from_output(
        self,
        generated_text: str,
        formatted_prompt: str,
    ) -> str:
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
        return self._strip_thinking_content(generated_text)

    def _strip_thinking_content(self, generated_text: str) -> str:
        """Strip reasoning blocks as a safety net for hybrid models."""
        if self.enable_thinking:
            return generated_text

        without_closed_blocks = re.sub(
            r"<think>.*?</think>",
            "",
            generated_text,
            flags=re.DOTALL,
        ).strip()
        if without_closed_blocks.startswith("<think>"):
            _, _, tail = without_closed_blocks.partition("</think>")
            return tail.strip() if tail else ""
        return without_closed_blocks

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text from a single prompt."""
        return self.generate_many(
            prompt,
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )[0]

    def generate_many(
        self,
        prompt: str,
        num_generations: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> List[str]:
        """Generate multiple sampled texts from the same prompt."""
        formatted_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs)

        gen_kwargs = self._build_generation_kwargs(
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        return [
            self._strip_prompt_from_output(
                self.tokenizer.decode(output, skip_special_tokens=True),
                formatted_prompt,
            )
            for output in outputs
        ]

    def batch_generate(
        self,
        prompts: List[str],
        num_generations: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        batch_size: int = 3,
        **kwargs,
    ) -> List[List[str]]:
        """Generate text for multiple prompts in batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        results: List[List[str]] = []
        batch_num_generations = num_generations or self.num_generations

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            formatted_batch = [self._format_prompt(prompt) for prompt in batch]
            inputs = self.tokenizer(
                formatted_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = self._move_inputs_to_device(inputs)
            gen_kwargs = self._build_generation_kwargs(
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            for j in range(len(batch)):
                start = j * batch_num_generations
                end = start + batch_num_generations
                prompt_outputs = outputs[start:end]
                results.append(
                    [
                        self._strip_prompt_from_output(
                            self.tokenizer.decode(output, skip_special_tokens=True),
                            formatted_batch[j],
                        )
                        for output in prompt_outputs
                    ]
                )

        return results

    def __repr__(self) -> str:
        return f"HFChatModel(model={self.model_name}, device={self.device})"


QwenModel = HFChatModel
