"""Example: Running inference with Qwen model on generated prompts using MPS."""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from indobias.models import QwenModel

def find_prompts_files(base_dir: Path) -> list[Path]:
    """Find all prompt files in the data directory structure."""
    data_dir = base_dir / "data"
    if not data_dir.exists():
        # Fallback to old structure
        old_file = base_dir / "prompts.jsonl"
        return [old_file] if old_file.exists() else []
    
    # Find all prompts.jsonl files in dimension subdirectories
    return sorted(data_dir.glob("*/prompts.jsonl"))

def load_all_prompts(prompt_files: list[Path]) -> list[dict]:
    """Load all prompts from multiple files."""
    all_prompts = []
    for prompt_file in prompt_files:
        with open(prompt_file, 'r') as f:
            for line in f:
                all_prompts.append(json.loads(line))
    return all_prompts

def main():
    # Setup paths
    output_dir = Path("output")
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Find prompt files
    prompt_files = find_prompts_files(output_dir)
    
    if not prompt_files:
        print("❌ No prompts found.")
        print("\nRun: uv run indobias generate --groups data/groups/ethnicity_indonesia.csv")
        return
    
    print(f"Found {len(prompt_files)} prompt file(s):")
    for pf in prompt_files:
        print(f"  • {pf.relative_to(output_dir)}")
    print()
    
    # Load all prompts
    print("Loading all prompts...")
    all_prompts = load_all_prompts(prompt_files)
    print(f"✓ Loaded {len(all_prompts)} total prompts\n")
    
    # Initialize model with MPS
    print("Initializing Qwen model with MPS...")
    model = QwenModel(
        model_name="Qwen/Qwen3-0.5B-Instruct",
        max_length=100,
        temperature=0.7
    )
    print()
    
    # Create run directory organized by groups file → dimension → task
    timestamp = datetime.now().strftime("%m-%d_%H%M%S")  # MM-DD_HHMMSS format
    run_dir = models_dir / f"qwen_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Running Inference - Output: {run_dir.relative_to(output_dir)}")
    print("=" * 80 + "\n")
    
    # Group prompts by groups file (demography type), dimension, and task
    prompts_by_groups_dim_task = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Determine groups file from metadata
    groups_files = set()
    for prompt_data in all_prompts:
        demography = prompt_data["metadata"]["demography"]
        dimension = prompt_data["metadata"]["dimension"]
        task_id = prompt_data["metadata"]["task_id"]
        
        # For ethnicity, use "ethnicity_indonesia"
        # For other demographies, use the demography name
        if demography == "ethnicity":
            groups_name = "ethnicity_indonesia"
        else:
            groups_name = demography
        
        groups_files.add(groups_name)
        prompts_by_groups_dim_task[groups_name][dimension][task_id].append(prompt_data)
    
    total_processed = 0
    result_id = 1  # Global ID counter
    
    # Process each groups file
    for groups_name in sorted(prompts_by_groups_dim_task.keys()):
        print(f"\n{'='*80}")
        print(f"📂 GROUPS: {groups_name.upper()}")
        print(f"{'='*80}")
        
        # Create groups directory
        groups_dir = run_dir / groups_name
        groups_dir.mkdir(parents=True, exist_ok=True)
        
        dimensions = prompts_by_groups_dim_task[groups_name]
        
        # Process each dimension within the groups file
        for dimension in sorted(dimensions.keys()):
            print(f"\n  📊 Dimension: {dimension.upper()}")
            print("  " + "-" * 76)
            
            # Create dimension directory
            dim_dir = groups_dir / dimension
            dim_dir.mkdir(parents=True, exist_ok=True)
            
            tasks = dimensions[dimension]
            
            # Process each task within the dimension
            for task_id in sorted(tasks.keys()):
                task_prompts = tasks[task_id]
                task_name = task_prompts[0]["metadata"]["task_name"]
                
                print(f"\n    📋 Task {task_id}: {task_name}")
                print(f"       Total prompts: {len(task_prompts)}")
                print("    " + "-" * 72)
                
                # Create task output file within dimension directory
                results_file = dim_dir / f"task_{task_id}_results.jsonl"
                
                # Run inference
                with open(results_file, 'w') as out_f:
                    for i, prompt_data in enumerate(task_prompts, 1):
                        prompt_text = prompt_data["prompt"]
                        metadata = prompt_data["metadata"]
                        
                        # Progress indicator (every 50 prompts)
                        if i % 50 == 0 or i == 1:
                            print(f"       Progress: {i}/{len(task_prompts)} ({i*100//len(task_prompts)}%)")
                        
                        # Generate response
                        try:
                            responses = model.generate_many(prompt_text, max_new_tokens=50)
                            
                            # Save result with ID
                            result = {
                                "id": result_id,
                                "groups_file": groups_name,
                                "prompt": prompt_text,
                                "response": responses[0],
                                "responses": responses,
                                "num_generations": len(responses),
                                "metadata": metadata,
                                "model": model.model_name,
                                "device": model.device,
                                "timestamp": datetime.now().isoformat()
                            }
                            out_f.write(json.dumps(result) + "\n")
                            total_processed += 1
                            result_id += 1
                            
                        except Exception as e:
                            print(f"       ❌ Error on prompt {i}: {e}")
                            # Log error with ID
                            result = {
                                "id": result_id,
                                "groups_file": groups_name,
                                "prompt": prompt_text,
                                "response": None,
                                "error": str(e),
                                "metadata": metadata,
                                "model": model.model_name,
                                "device": model.device,
                                "timestamp": datetime.now().isoformat()
                            }
                            out_f.write(json.dumps(result) + "\n")
                            result_id += 1
                
                print(f"       ✓ Completed: {len(task_prompts)} prompts")
                print(f"       ✓ Saved to: {results_file.relative_to(output_dir)}")
    
    print("\n" + "=" * 80)
    print("✓ Inference Complete!")
    print(f"  Total prompts processed: {total_processed}")
    print(f"  Groups files: {len(groups_files)}")
    print(f"  Dimensions per group: {len(prompts_by_groups_dim_task[list(groups_files)[0]])}")
    print("  Tasks per dimension: 7")
    print(f"  Model: {model.model_name}")
    print(f"  Device: {model.device}")
    print(f"  Results saved to: {run_dir.relative_to(output_dir)}/")
    print("=" * 80)

if __name__ == "__main__":
    main()
