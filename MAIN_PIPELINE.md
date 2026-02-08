# IndoBias Main Pipeline

This document explains the simplified `main.py` and how it integrates with the IndoBias framework.

## Architecture

The `main.py` script is now much simpler and directly uses the core IndoBias modules:

```
main.py
├── Uses: indobias.generator (generate_prompts, load_dimension_catalog, load_groups_from_csv)
├── Uses: indobias.data_models (GroupCollection, Demography, StereotypeDimension)
├── Uses: indobias.tasks (ALL_TASKS, get_task)
└── Uses: indobias.models.* (QwenModel, OpenAIModel, etc.)
```

## Key Features

### 1. Direct Module Integration
- **No subprocess calls** - Uses Python imports directly
- **Type-safe** - Leverages Pydantic models from `data_models.py`
- **Clean code** - Much shorter and easier to maintain

### 2. Customizable Command-Line Arguments

```bash
# Run for all groups (default)
python main.py

# Specific groups
python main.py --groups ethnicity_indonesia gender

# Different model
python main.py --model qwen --model-id Qwen/Qwen2.5-3B-Instruct

# Skip generation or inference
python main.py --skip-generation
python main.py --generate-only

# List available options
python main.py --list-groups
python main.py --list-models
```

### 3. Model Backend System

The `load_model()` function makes it easy to add new models:

```python
def load_model(model_name: str, model_id: str, **kwargs):
    if model_name == "qwen":
        from indobias.models import QwenModel
        return QwenModel(model_name=model_id, **kwargs)
    elif model_name == "openai":
        from indobias.models.openai import OpenAIModel
        return OpenAIModel(model_name=model_id, **kwargs)
    # ... add more models here
```

**Currently implemented:**
- ✅ `qwen` - Local Qwen models with MPS/CUDA support

**Template files created:**
- ⏳ `indobias/models/openai.py` - OpenAI API (TODO)
- ⏳ `indobias/models/anthropic.py` - Anthropic Claude API (TODO)
- ⏳ `indobias/models/gemini.py` - Google Gemini API (TODO)

### 4. Two-Step Pipeline

#### Step 1: Generate Prompts
- Loads dimension catalog from `data/dimensions.yaml`
- Loads groups from `data/groups/*.csv`
- Generates prompts using `generate_prompts()` from `indobias.generator`
- Exports to `output/data/{dimension}/prompts.jsonl`

#### Step 2: Run Inference
- Loads all generated prompts
- Initializes specified model
- Runs inference with progress tracking
- Saves results to `output/models/{model}_{timestamp}/`

### 5. Output Structure

```
output/
├── data/
│   ├── wgi/
│   │   └── prompts.jsonl
│   ├── onet/
│   │   └── prompts.jsonl
│   ├── qs_rankings/
│   │   └── prompts.jsonl
│   └── sp/
│       └── prompts.jsonl
└── models/
    └── qwen_qwen2_5_0_5b_instruct_02-08_143022/
        ├── ethnicity/
        │   ├── sp/
        │   │   ├── task_1_results.jsonl
        │   │   ├── task_2_results.jsonl
        │   │   └── ...
        ├── gender/
        ├── religion/
        └── ...
```

## Adding a New Model

1. **Create model file**: `indobias/models/<model_name>.py`

2. **Implement the model class**:
```python
class MyModel:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        # Initialize your model here
    
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        # Implement generation logic
        return "generated text"
```

3. **Add to load_model()** in `main.py`:
```python
elif model_name == "mymodel":
    from indobias.models.mymodel import MyModel
    return MyModel(model_name=model_id, **kwargs)
```

4. **Update choices** in argument parser:
```python
parser.add_argument(
    '--model',
    choices=['qwen', 'openai', 'anthropic', 'gemini', 'mymodel'],
    # ...
)
```

## Benefits Over Previous Version

### Before (515 lines):
- ❌ Used subprocess to call CLI commands
- ❌ Complex string manipulation for commands
- ❌ Harder to debug
- ❌ More error-prone

### After (405 lines):
- ✅ Direct Python imports
- ✅ Type-safe with Pydantic models
- ✅ Easy to debug
- ✅ Clean and maintainable
- ✅ Better progress reporting with rich
- ✅ Modular and extensible

## Quick Start

```bash
# Install dependencies
pip install -e .

# List available groups
python main.py --list-groups

# Run for specific groups
python main.py --groups ethnicity_indonesia gender

# Run full pipeline
python main.py
```

## Integration with Existing Code

The script works seamlessly with existing modules:
- ✅ `indobias/cli.py` - CLI commands still work independently
- ✅ `indobias/generator.py` - Used directly for generation
- ✅ `indobias/data_models.py` - All data structures reused
- ✅ `indobias/tasks.py` - Task templates from the framework
- ✅ `indobias/models/qwen.py` - Model implementation reused
