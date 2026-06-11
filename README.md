# Euro_sat_v2

## Project Structure

```
Euro_sat_v2/
├── config.yaml          # Centralized hyperparameters and settings
├── train.py             # Main pipeline to execute model training
├── gui_predict.py       # Desktop GUI for drag-and-drop image lookup
├── resource.py          # Windows compatibility mock file
├── checkpoints/         # Generated automatically (stores best model weights)
│   └── best_model.pth
└── src/                 # Core modular source code
    ├── __init__.py
    ├── dataset.py       # Handles TFDS data streaming and PyTorch loaders
    └── model.py         # Defines the Convolutional Neural Network layers
```

## Why the Code is Split Up (Software Architecture)

This project has been refactored out of a monolithic Jupyter Notebook and into a modular, production-grade Python structure based on the Separation of Concerns principle.

Here is why this modular structure makes the project significantly more robust:

### 1. Isolated Responsibilities

Each file in the `src/` directory has exactly one job:

- **dataset.py** handles the data plumbing. If you decide to transition from streaming via TFDS to loading local GeoTIFF files or an Amazon S3 bucket, you only change this file. The model and training logic remain untouched.

- **model.py** handles the network architecture. If you want to replace the current custom CNN with a pre-trained Vision Transformer (ViT) or ResNet, you modify this file alone.

### 2. Elimination of Global State Chaos

Jupyter Notebooks store variables in a single global memory space where the cell execution order matters. If cells are run out of sequence, variables can spill over and corrupt model states. Modularization enforces clear boundaries between components, preventing these kinds of hidden dependencies.

### 3. True Code Reusability (The GUI Benefit)

In the original notebook layout, testing a single new satellite picture required executing the entire notebook from top to bottom, including downloading datasets and retraining.

By separating the logic, both the heavy automated training script (`train.py`) and the lightweight interactive interface (`gui_predict.py`) effortlessly import and share the exact same neural network blueprint:

```python
from src.model import EuroSATCNN
```

This eliminates redundant code duplication and allows the GUI to spin up instantly using pre-trained checkpoint weights.

## Getting Started

### 1. Prerequisites

Ensure you have Anaconda or Python installed, then install the required dependencies:

```bash
pip install torch torchvision tensorflow-datasets pyyaml pillow numpy
```

**Note for Windows Users:** If TensorFlow Datasets complains about a missing system module, the repository includes a local `resource.py` file to mock Unix compatibility safely.

### 2. Manual Data Staging (Server 403 Bypass)

Because the official remote EuroSAT hosting server blocks automated script downloads, you must stage the dataset file manually:

1. Download the file directly via your browser from the official archive: [http://madm.dfki.de/files/sentinel/EuroSAT.zip](http://madm.dfki.de/files/sentinel/EuroSAT.zip)

2. Do not extract it. Leave it zipped and place it exactly in this directory:
   ```
   C:\Users\<YOUR_USERNAME>\tensorflow_datasets\downloads\manual\EuroSAT.zip
   ```

### 3. Training the Model

To start the automated training loop, run the following command in your terminal:

```bash
python train.py
```

The script will stream the arrays from your manual zip cache, apply random vertical/horizontal flip augmentations, and save the highest-performing model state inside a new `./checkpoints/` directory.

### 4. Running the Interactive Lookup GUI

Once you have generated a `best_model.pth` file from training, launch the visual desktop application:

```bash
python gui_predict.py
```

- Click "Choose Satellite Picture" to open your native file explorer
- Select any satellite snapshot or image tile
- The application will upsample the image to a readable 256x256 panel for display, pass a downscaled 64x64 normalized tensor to the PyTorch engine, and display the predicted category along with the model's confidence score

---

**For questions or issues, please open a GitHub issue in this repository.**
