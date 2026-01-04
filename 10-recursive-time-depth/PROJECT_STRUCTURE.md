# Project Structure

## Main Files

### Code
- **`recursive_time_depth.py`** - Main experiment file
  - Implementation of recursive forward with adaptive stopping
  - Support for different layers for recursion
  - Condensation metrics (effective rank, SVCCA)
  - Fixed entropy
  
- **`strict_validation_tests.py`** - 5 strict validation tests
  - Test 1: Distribution of relative_change per token
  - Test 2: Different layers for recursion
  - Test 3: Text variations (normal vs shuffle vs random)
  - Test 4: Accuracy comparison
  - Test 5: Fixed entropy

### Documentation
- **`README.md`** - Main project documentation
- **`TEST_RESULTS_FINAL.md`** - Final test results and analysis
- **`PROJECT_STRUCTURE.md`** - This file

### Configuration
- **`requirements.txt`** - Project dependencies
- **`.gitignore`** - Ignored files

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run main experiment: `python recursive_time_depth.py`
3. Run strict validation tests: `python strict_validation_tests.py`
4. Read results: `TEST_RESULTS_FINAL.md`

## Key Features

- ✅ Fixed attention_mask handling
- ✅ Recursion on subnetworks (not entire model)
- ✅ Percentiles with attention_mask consideration
- ✅ Support for different layers for recursion
- ✅ Strict validation tests
- ✅ Correct formulations of conclusions
