# Scripts Directory 📁

This directory contains organized scripts for PainCare AI model management.

## 📁 Directory Structure

### `/training/`
Contains scripts for training AI models with real data.

- **`enhanced_real_data_trainer.py`** - Main training script for PainCare AI models
  - Uses real user data from Firebase
  - Implements evidence-based training patterns
  - Trains multiple models: pain predictor, treatment recommender, symptom analyzer
  - Includes model validation and performance metrics

### `/data_generation/`
Contains scripts for generating synthetic test data (development use only).

- **`generate_realistic_dataset.py`** - Generates synthetic endometriosis data
  - Creates medically plausible test data
  - Useful for development and testing
  - Generates CSV and JSON formats

## 🚀 Usage

### Training Models
```bash
# Train AI models with real data
python scripts/training/enhanced_real_data_trainer.py
```

### Generating Test Data (Development Only)
```bash
# Generate synthetic data for testing
python scripts/data_generation/generate_realistic_dataset.py
```

## 📝 Notes

- **Production Training**: Use only `enhanced_real_data_trainer.py` for production models
- **Synthetic Data**: Generated data is for development/testing only
- **Real Data**: All production models are trained on real user data from Firebase
- **Model Output**: Trained models are saved to `/models/` directory

## 🧹 Cleaned Items

Removed during workspace cleanup:
- ❌ `train_with_realistic_data.py` (redundant)
- ❌ `/generated_data/` directory (8.19 MB of synthetic test data)
- ❌ Various CSV/JSON synthetic data files

## ✅ Current Status

- **Organized**: Scripts properly categorized by function
- **Clean**: Redundant files removed
- **Documented**: Clear usage instructions
- **Production Ready**: Real data training script available
