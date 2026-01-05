# Push Instructions

## Files to Add

### New Files (untracked)
```bash
git add .gitignore
git add FATIGUE_TEST_RESULTS.md
git add HYSTERESIS_METRICS.md
git add HYSTERESIS_TEST_README.md
git add METRICS_DEFINITIONS.md
git add test_fatigue.py
git add test_hysteresis.py
git add fatigue_analysis.png
git add hysteresis_analysis.png
git add hysteresis_results.json
```

### Modified Files
```bash
git add README.md
git add requirements.txt
git add temporal_lora.py
```

## Files Excluded (via .gitignore)
- `__pycache__/` - Python cache
- `temporal_lora_checkpoint.pt` - Large model checkpoint (users should train their own)
- `*.pyc` - Compiled Python files

## Commit Message Suggestion

```
feat: Add hysteresis and fatigue tests for router time crystallization

- Add hysteresis test (A->B->A, A->Mix->A) with switch-lag and return-gap metrics
- Add fatigue test (deep crystallization) with varying domain stay length
- Add comprehensive documentation (METRICS_DEFINITIONS.md, FATIGUE_TEST_RESULTS.md)
- Translate all code and documentation to English
- Update README with new experiments and results
- Add .gitignore for Python projects
```

## Push Command

```bash
git commit -m "feat: Add hysteresis and fatigue tests for router time crystallization"
git push origin main
```

