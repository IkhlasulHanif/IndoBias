"""Qwen model implementation with MPS support for Apple Silicon."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Any


class QwenModel:
    """Qwen model with MPS (Metal Performance Shaders) support for Apple Silicon Macs."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: Optional[str] = None,
        max_length: int = 2048,
        num_generations: int = 10,
        temperature: float = 0.9,
        top_p: float = 0.9,
    ):
        """
        Initialize Qwen model with MPS support.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto-detect)
            max_length: Maximum generation length
            num_generations: Default number of responses to sample per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_generations = num_generations
        self.temperature = temperature
        self.top_p = top_p
        self.chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Loading {model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        if self.device == "mps":
            # For MPS, load on CPU first then move to MPS
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
        else:
            # For CUDA/CPU, use device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=self.device if self.device == "cuda" else "auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        print(f"✓ Model loaded successfully on {self.device}")

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
        return {
            "max_new_tokens": max_new_tokens or self.max_length,
            "num_return_sequences": num_generations,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }

    def _move_inputs_to_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _strip_prompt_from_output(self, generated_text: str, formatted_prompt: str) -> str:
        if generated_text.startswith(formatted_prompt):
            return generated_text[len(formatted_prompt):].strip()
        return generated_text
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a single prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate (overrides max_length)
            temperature: Sampling temperature (overrides default)
            top_p: Nucleus sampling (overrides default)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
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
        """
        Generate multiple sampled texts from the same prompt.

        Args:
            prompt: Input prompt text
            num_generations: Number of responses to sample
            max_new_tokens: Maximum tokens to generate (overrides max_length)
            temperature: Sampling temperature (overrides default)
            top_p: Nucleus sampling (overrides default)
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        # Prepare input using a shared chat-template configuration.
        formatted_prompt = self._format_prompt(prompt)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Move inputs to device
        inputs = self._move_inputs_to_device(inputs)
        
        # Set generation parameters
        gen_kwargs = self._build_generation_kwargs(
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
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
        batch_size: int = 4,
        **kwargs
    ) -> List[List[str]]:
        """
        Generate text for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            num_generations: Number of responses to sample per prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            batch_size: Batch size for processing
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated-text lists, one per prompt
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Apply chat template to each prompt in batch.
            formatted_batch = [self._format_prompt(prompt) for prompt in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                formatted_batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            inputs = self._move_inputs_to_device(inputs)
            
            # Generate
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
                    **gen_kwargs
                )
            
            batch_num_generations = num_generations or self.num_generations

            # Decode batch
            for j in range(len(batch)):
                start = j * batch_num_generations
                end = start + batch_num_generations
                prompt_outputs = outputs[start:end]
                results.append([
                    self._strip_prompt_from_output(
                        self.tokenizer.decode(output, skip_special_tokens=True),
                        formatted_batch[j],
                    )
                    for output in prompt_outputs
                ])
        
        return results
    
    def __repr__(self) -> str:
        return f"QwenModel(model={self.model_name}, device={self.device})"


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
        "cpu_available": True,
    }
    
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    
    return info


if __name__ == "__main__":
    # Test device info
    print("Device Information:")
    info = get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test model loading
    print("\nTesting Qwen model with MPS...")
    model = QwenModel(model_name="Qwen/Qwen2.5-0.5B-Instruct")  # Smaller model for testing
    
    # Test generation
    test_prompt = "What is the capital of Indonesia?"
    print(f"\nPrompt: {test_prompt}")
    response = model.generate(test_prompt, max_new_tokens=50)
    print(f"Response: {response}")
