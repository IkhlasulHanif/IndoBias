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
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize Qwen model with MPS support.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto-detect)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
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
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate (overrides max_length)
            temperature: Sampling temperature (overrides default)
            top_p: Nucleus sampling (overrides default)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to device
        if self.device == "mps":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_length,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove the prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate response using chat format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{"role": "user", "content": "Hello"}]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response text
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            batch_size: Batch size for processing
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated texts
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            if self.device == "mps":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            gen_kwargs = {
                "max_new_tokens": max_new_tokens or self.max_length,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # Decode batch
            for j, output in enumerate(outputs):
                generated = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove prompt
                if generated.startswith(batch[j]):
                    generated = generated[len(batch[j]):].strip()
                results.append(generated)
        
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
