# Amazon Reviews Classification

A machine‐learning / data‐analysis project exploring ecommerce event classification and confidence evaluation using the Amazon e­Commerce dataset.

## 🚀 Project Overview  
This repository contains code, experiments, and evaluation assets for the “Amazon ECEN” project:  
- A Jupyter notebook (`deberta_training.ipynb`) that steps through data loading, preprocessing, model training, confidence distribution analysis, and error‐investigation.
- Unit / functional test code (`test.py`) to validate key pipeline components.
- A full evaluation results breakdown including confusion matrices, F1‐score plots, and metrics comparisons (`confusion_matrix.png`, `f1_scores.png`, `metrics_comparison.png`, etc).
- A requirements file (`requirements.txt`) listing Python dependencies for reproducibility.
- Sample data input (`test.csv`) and output logs (`evaluation_results.txt`).

## 📂 Repository Structure  
| File Structure |
| :--- |
| `best_lstm_model.pt` |
| `best_rnn_model.pt` |
| `data_preprocessing_and_eda.py` |
| `deberta_training.py` |
| `evaluation_results.txt` |
| `README.md` |
| `requirements.txt` |
| `rnn_lstm_training.py` |
| `test.csv` |
| `test.py` |
| `train.csv` |
| `evaluation_results.txt` |
| `confusion_matrix.png` |
| `f1_scores.png` |
| `metrics_comparison.png` |
| `confidence_distribution.png` |
| `correct_incorrect_by_confidence.png`|
| `epoch2` |
| `comparison_all_models.png` |
| `.gitattributes` |
| `.DS_Store` |

## 🧠 Key Features  
- **Data Preprocessing and EDA**: pre-process and analyze required amount of data by running `data_preprocessing_and_eda.py`
- **End‐to‐end ML workflow**: from dataset ingestion → preprocessing → training → evaluation.  
- **Confidence‐based error analysis**: plot distributions of prediction confidence and investigate misclassifications vs confidence thresholds (`confidence_distribution.png`, `correct_incorrect_by_confidence.png`).  
- **Comprehensive evaluation**: visualize confusion matrices, F1 scores over classes, and compare metrics across model variants (`metrics_comparison.png`).  
- **Reproducible setup**: easily install dependencies via `requirements.txt` and run `test.py` to verify core functionality.  
- **Experiment notebook**: the Jupyter notebook contains documented steps, code explanations, visualizations, and analysis commentary.
- **Comparison of all Experimented Models**: view the metric comparison graph for all experimented models in `comparison_all_models.png`

## 🛠️ Getting Started  
### Prerequisites  
- Python 3.8+ (or your preferred version)  
- Virtual environment recommended (e.g., `venv`, `conda`)  
- Install dependencies:  
  ```bash
    pip install -r requirements.txt
  ```
-  Clone the repository with Git LFS 
  - Install Git LFS if not already installed
  ```bash                                                                                                                                                                                                                                                                      
    git lfs install
  ```
  - Clone the repository (this will download LFS files automatically)                                                                                                                                                                                                                                             
  ```bash
    git clone <repository-url>
  ```
### Running the notebook
- Launch Jupyter or open deberta_training.ipynb in your IDE.

- Run the cells sequentially: dataset loading → preprocessing → training → evaluation → analysis.

- Inspect the visual outputs: confidence distribution, confusion matrix, F1-scores.

- Running tests
  ```bash
    python test.py
  ```
This executes core functional tests to ensure pipeline integrity (data loading, preprocessing, prediction outputs, etc).

## 📊 Results
- Confidence distribution: confidence_distribution.png illustrates how prediction confidence correlates with correct vs incorrect outputs.

- Confusion matrix: confusion_matrix.png shows classification performance per class.

- F1-scores and comparison: f1_scores.png and metrics_comparison.png allow side‐by‐side comparison of model variants and class‐level performance.

- Correct vs incorrect by confidence: correct_incorrect_by_confidence.png helps identify high-confidence misclassifications for deeper error‐analysis.

## 💡 Use Cases
- Use this project as a reference workflow for event classification or confidence‐based analysis in classification models.

- Extend and adapt the notebook to new datasets (e.g., other ecommerce logs, user interaction data).

- Plug in your own model architectures, thresholds, or error-analysis strategies based on the existing evaluation framework.

## 🗂️ Future Work
- Expand dataset size and diversity (e.g., full Amazon e-commerce logs, additional features).

- Experiment with advanced architectures (transformers, sequence models) for event classification.

- Implement automatic threshold tuning for confidence‐based decision‐making.

- Add model serialization, deployment scripts and live inference demo.

- Integrate dataset versioning and pipeline automation (CI/CD, scheduled runs, etc).

## 📄 License
- This repository is provided under the MIT License – see the LICENSE file for details.
