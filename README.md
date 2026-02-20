# STL10-Image-Classifiers

This project implements a complete deep learning pipeline for classifying images from the STL-10 dataset using various architectures (Logistic Regression, Fully Connected, CNNs, and MobileNet). The pipeline includes data augmentation, training with validation splitting, and a comprehensive evaluation suite.

## 🚀 Quick Start (How to Run)

Follow these steps in order to easily run the project from start to finish:

1. **Get the Data**: Download the STL-10 dataset (~2.5GB) and prepare DataLoaders.
   ```bash
   python get_data.py
   ```
2. **Visualize (Optional)**: Generate augmentation samples and class grids. Results are saved in `plots and outputs/`.
   ```bash
   python plot_data_sample.py
   ```
3. **Configure Model**: Open `config.py` and set your desired `MODEL_TYPE`. Available options:
   * `'logistic'`: Basic Logistic Regression.
   * `'fc'`: Fully Connected (FC) Neural Network (MLP).
   * `'cnn'`: Simple Convolutional Neural Network.
   * `'mobilenet_fixed'`: MobileNetV2 as a fixed feature extractor (only the head is trained).
   * `'mobilenet_learned'`: MobileNetV2 with full fine-tuning (all weights are updated).

4. **Train**: Run the training script. The best model and curves will be saved in `plots and outputs/`.
   ```bash
   python train.py
   ```
5. **Evaluate**: Test the model on unseen data. This generates a classification report and a confusion matrix in `plots and outputs/`.
   ```bash
   python test.py
   ```

## 🛠️ Installation & Setup

### 1. Fix Windows DLL Errors
To prevent `DLL load failed` on Windows:
* Install **Microsoft Visual C++ Redistributable (x64)**: [Download here](https://aka.ms/vs/17/release/vc_redist.x64.exe)
* **Restart your computer** after installation.

### 2. Virtual Environment & Dependencies
**Choose the installation command based on your hardware:**

**Option A: For standard laptops (CPU-only)**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt
```

**Option B: For machines with NVIDIA GPU (CUDA)**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 📁 Project Structure

* `config.py`: Centralized configuration for hyperparameters and model selection.
* `get_data.py`: Data loading logic and custom augmentation wrappers.
* `train.py`: Main training loop; saves results to `plots and outputs/`.
* `test.py`: Evaluation script; loads weights and generates metrics.
* `plot_data_sample.py`: Tool for inspecting dataset and augmentations.
* `model_*.py` & `mobilenet.py`: Architecture definitions for all classifiers.
* `plots and outputs/`: (Auto-generated) Contains saved models (.pth) and performance plots (.png).
