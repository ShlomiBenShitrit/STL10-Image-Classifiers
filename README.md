# STL-10 Image Classification Pipeline

This project implements a complete deep learning pipeline for classifying images from the STL-10 dataset using various architectures (Logistic Regression, Fully Connected, CNNs, and MobileNet). The pipeline includes data augmentation, training with validation splitting, and a comprehensive evaluation suite with metrics and confusion matrices.

## 🚀 Quick Start (How to Run)

Follow these steps in order to easily run the project from start to finish:

1. **Get the Data**: Download the STL-10 dataset (~2.5GB) and prepare DataLoaders.
   ```bash
   python get_data.py
   ```
2. **Visualize (Optional)**: Generate augmentation samples and class grids to verify data.
   ```bash
   python plot_data_sample.py
   ```
3. **Configure Model**: Open `config.py` and set your desired `MODEL_TYPE` (e.g., `'logistic'`, `'cnn'`, `'mobilenet_learned'`).
4. **Train**: Run the training script. The model with the best validation accuracy will be saved automatically.
   ```bash
   python train.py
   ```
5. **Evaluate**: Test the model on unseen data. This will print a classification report and generate a confusion matrix plot.
   ```bash
   python test.py
   ```

## 🛠️ Installation & Setup

### 1. Fix Windows DLL Errors
To prevent `DLL load failed` or `OSError: [WinError 1114]` when running PyTorch on Windows:
* Install the **Microsoft Visual C++ Redistributable (x64)**: [Download here](https://aka.ms/vs/17/release/vc_redist.x64.exe)
* **Restart your computer** after installation.

### 2. Virtual Environment & Dependencies
Clone the repository, create a virtual environment, and install the required packages. 
**Choose the installation command based on your hardware:**

**Option A: For machines WITHOUT a dedicated GPU (e.g., standard laptops)**
To avoid large downloads and CUDA errors, install the CPU-optimized version of PyTorch first:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt
```

**Option B: For machines WITH an NVIDIA GPU (or servers)**
Install directly from the requirements file to automatically get the full CUDA-enabled PyTorch version:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 📁 Project Structure

* `config.py`: Centralized configuration for hyperparameters, model selection (`MODEL_TYPE`), and paths.
* `get_data.py`: Data loading logic, including custom `DatasetWithTransform` wrapper for training/validation split and data augmentation.
* `train.py`: Main training loop with model saving and reproducibility settings.
* `test.py`: Evaluation script that generates a classification report and a confusion matrix plot.
* `plot_data_sample.py`: Visualization tool for inspecting the dataset and augmentations.
* `model_cnn.py`, `model_fc.py`, `model_logistic.py`, `mobilenet.py`: Individual files containing the neural network architecture definitions.
* `requirements.txt`: List of Python dependencies required to run the project.

## 📊 Features & Results

* **Dynamic Checkpoints**: To prevent different architectures from overwriting each other, outputs are saved dynamically using the model's name (e.g., `best_model_cnn.pth`, `training_curves_cnn.png`).
* **Confusion Matrices**: After running `test.py`, a file like `confusion_matrix_logistic.png` is generated in the root folder, showing exactly where the model is confusing classes (e.g., Cat vs. Dog).
* **Hardware Auto-Detection**: The code automatically uses CUDA (GPU) if available for faster training, but safely falls back to CPU (using `map_location`) for local testing and inference.