# Dataset Compatibility Guide

## Overview

The MoE Research Hub has been enhanced with robust dataset validation and loading capabilities. This guide explains which datasets are compatible and how to use them.

## Compatible Dataset Requirements

### Hugging Face Datasets

A Hugging Face dataset is compatible if it meets these requirements:

1. **Required Splits**: Must have a `train` split
2. **Optional Splits**: May have `validation`, `test`, or `dev` splits (if missing, train will be auto-split 90/10)
3. **Field Structure**: Must have either:
   - A `text` field containing the text data, OR
   - Both `question` and `answer` fields (with optional `reasoning` field)

### Local Files

Local files are supported in these formats:
- **`.txt`**: One sample per line
- **`.json/.jsonl`**: JSON objects with `text` field or `question`/`answer`/`reasoning` fields

## Confirmed Compatible Datasets

### Text Datasets (have `text` field)
- ✅ `wikitext/wikitext-2-v1` - Wikipedia text
- ✅ `wikitext/wikitext-103-v1` - Larger Wikipedia corpus  
- ✅ `imdb` - Movie reviews
- ✅ `openwebtext` - Web text corpus
- ✅ `bookcorpus` - Book text
- ✅ `c4` - Cleaned Common Crawl

### Question-Answer Datasets
- ✅ Local lambda calculus dataset (your format)
- ✅ Any dataset with `question`/`answer` structure

## Incompatible Datasets

### Common Issues
- ❌ `squad` - Has `answers` (plural) instead of `answer` (singular)
- ❌ `glue/*` - Various field structures not matching our requirements
- ❌ Datasets with only `context`, `premise`, `hypothesis` fields
- ❌ Datasets without text content

## Validation Process

The system automatically validates datasets before loading:

1. **Existence Check**: Verifies dataset exists on HuggingFace Hub
2. **Split Validation**: Checks for required train split
3. **Structure Validation**: Verifies compatible field structure
4. **Sample Testing**: Tests loading a small sample to ensure compatibility

## Error Handling

The system now uses **fail-fast error handling** with no synthetic data fallback. When errors occur:

### Dataset Validation Errors
```
❌ Dataset validation failed: Dataset must have either a 'text' field or 'question'/'answer' fields. Available fields: ['id', 'title', 'context', 'question', 'answers']

📋 Compatible dataset requirements:
   - Must have a 'train' split
   - Must have either a 'text' field OR 'question'/'answer' fields
   - Recommended datasets: wikitext, openwebtext, bookcorpus, c4
```

### File Not Found Errors
```
FileNotFoundError: Local file not found: datasets/missing_file.json
```

### JSON Format Errors
```
ValueError: Invalid JSON format in file datasets/corrupt.json: Expecting ',' delimiter: line 5 column 10 (char 89)
```

### Empty Dataset Errors
```
ValueError: No valid text data found in file: datasets/empty.json
```

**Important**: The system no longer falls back to synthetic data. All errors are surfaced immediately so you can fix the underlying issue.

## Converting Incompatible Datasets

If you have a dataset that doesn't meet the requirements, you can easily convert it:

```python
# Example: Convert SQuAD format to compatible format
import json

def convert_squad_to_compatible(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    converted = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                converted.append({
                    'question': qa['question'],
                    'answer': qa['answers'][0]['text'],  # Take first answer
                    'context': paragraph['context']  # Optional
                })
    
    with open(output_file, 'w') as f:
        json.dump(converted, f)
```

## Best Practices

1. **Start with recommended datasets** like wikitext for initial experiments
2. **Test dataset compatibility** before large experiments using the validation
3. **Keep datasets focused** on text content rather than complex structured data
4. **Use local files** for custom datasets that require specific preprocessing

## Testing Dataset Compatibility

You can test any dataset before use:

```python
from core.data import validate_hf_dataset

is_valid, error_msg, info = validate_hf_dataset('dataset_name', 'config_name')
if is_valid:
    print(f"✅ Compatible: {info}")
else:
    print(f"❌ Incompatible: {error_msg}")
