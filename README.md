## Car with Machine Learning – Intelligent Driving Agent

This project implements **intelligent driving agents for a car in circuits**, combining **manual control with lane-centering assist**, **automatic PID control**, and **supervised machine learning models** (classification and regression).

---

## Project Structure

- **`coche_inteligente.py`**  
  Main program containing all agents and simulation modes:  
  - `ManualCar` → Manual driving mode, controlled with keyboard but with **lane-centering correction** (not fully manual). This assist makes driving easier by automatically applying small corrections to keep the car in the lane.  
  - `AutoCar` → Autonomous agent based on **PID control** (rules + sensor balancing).  
  - `SmartCar` → Intelligent agent that loads **trained classification and regression models** (`joblib`) to predict steering.

- **`clasificacion.py`**  
  Evaluates and compares different classification algorithms with encoders.  
  - Models: Decision Tree, Random Forest, KNN (k=3, 5), Naive Bayes, ZeroR.  
  - Encoders: Label, Ordinal, OneHot.

- **`regresion.py`**  
  Evaluates regression algorithms on the driving dataset.  
  - Models: Linear Regression, Decision Tree, Random Forest, KNN (k=3, 5), Gradient Boosting, ZeroR.  
  - Metric: **RMSE** (Root Mean Squared Error).

- **`clasificador_doble.py`**  
  Extended classifier that can combine or compare multiple classification strategies.

- **`train_model.py`**  
  Script for training and exporting models (`joblib`) for use in the intelligent agent.

- **Assets and Folders**  
  - `car.png` → Car sprite used in the simulation.  
  - `maps_procesados/` → Folder containing processed maps for the car to drive on.  
  - `test/` → Testing resources.  
  - `csv_classification_files/` → Intermediate CSV files generated during training.  
  - `modelo_clasificador.joblib` → Final classification model used by SmartCar.  
  - `modelo_regresion.joblib` → Regression model used by SmartCar.

---

## Driving Modes

You can select the mode in `coche_inteligente.py` by setting:

```python
MANUAL_MODE = True     # Keyboard controlled (with lane-centering assist)
AUTO_MODE   = True     # Autonomous (PID)
IA_MODE     = True     # Smart agent with ML models

### 1. Manual Mode (with lane-centering assist)

Drive with the keyboard, assisted by lane-centering correction.  
The car is not fully manual: it applies a small automatic correction to stay centered in the lane, making it easier to control.

- ⬅️ Left / A → Steer left
- ➡️ Right / D → Steer right
- R → Reset car
- S → Toggle sampling

Data is saved in `driving_data_manual.arff` and regression data in `driving_regression_manual_auto.arff`.

### 2. Automatic Mode

The car drives using PID control:

- Adjusts steering using radar sensors.
- Controls speed depending on front distance.
- Saves data in the same format as manual mode for consistency.

### 3. Intelligent Agent Mode

The car drives using trained ML models:

- **Classification model**: predicts steering classes (left, right, straight).
- **Regression model**: predicts continuous steering angles.
- Final steering = average of classifier and regressor predictions.

---

## How to Run

### Manual Driving (with assist)

```bash
python coche_inteligente.py   # with MANUAL_MODE=True
```

### Automatic Driving (PID)

```bash
python coche_inteligente.py   # with AUTO_MODE=True
```

### Intelligent Driving (ML)

Train models first with `clasificador_doble.py`, `regresion.py` or `train_model.py`, then run:

```bash
python coche_inteligente.py   # with IA_MODE=True
```

---

##  Workflow

### 1) Data Collection

Decide what type of data you want to collect:  
- **Classification** → generates `driving_data_auto_manual.arff`
- **Regression** → generates `driving_regression_manual_auto.arff`

Activate **Manual** or **Auto** mode in `coche_inteligente.py` and run:

```bash
python3 coche_inteligente.py
```

### 2A) Train CLASSIFIER

**Option 1: Direct training**

```bash
python3 clasificador_doble.py driving_data_auto_manual.arff
# outputs modelo_clasificador.joblib
```

**Option 2: Evaluate several models before exporting the best one**

```bash
python3 clasificacion.py train.arff test.arff
python3 train_model.py driving_data_auto_manual.arff modelo_clasificador.joblib
```

### 2B) Train REGRESSOR

**Option 1: Direct training**

```bash
python3 train_model.py driving_regression_manual_auto.arff modelo_regresion.joblib
```

**Option 2: Evaluate several regressors first**

```bash
python3 regresion.py train.arff test.arff
python3 train_model.py driving_regression_manual_auto.arff modelo_regresion.joblib
```

### 3) Drive with AI (SmartCar)

Make sure the following files exist:

- `modelo_clasificador.joblib`
- `modelo_regresion.joblib`

Set in `coche_inteligente.py`:

```python
MANUAL_MODE = False
AUTO_MODE   = False
IA_MODE     = True
```

Run:

```bash
python3 coche_inteligente.py
```

The SmartCar will load both models and combine classifier and regressor predictions to decide the steering.

If you want, you can create a decision tree to better understand how the agent is working.

---

## Designing Maps

You can design your own circuits using any drawing program (e.g., GIMP, Photoshop, Krita).

- The map image must have a resolution of 1920x1080.
- The track should be black.
- Add a red line to indicate the starting position.
- The program detects this red line to automatically place the car at the start.

---

##  Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install numpy pandas scikit-learn joblib matplotlib scipy opencv-python pygame
```
