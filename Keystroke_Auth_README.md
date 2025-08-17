
# Keystroke Authentication System

## Overview
The Keystroke Authentication System is a biometric authentication method that uses the unique typing patterns of individuals to verify their identity. This system analyzes various aspects of an individual's typing dynamics, such as key press timings, dwell time, cadence, and pauses, to create a personalized authentication model.

The project includes:
- **Keystroke capture**: Captures key events (keydown/keyup) and their timestamps.
- **Feature extraction**: Extracts relevant features like dwell time, key press intervals, cadence, and pauses.
- **Model training**: Utilizes machine learning algorithms (SGDClassifier) to train a model on the user's typing patterns.
- **KBA fallback**: If the model is not confident, a set of predefined security questions (KBA) are asked as a fallback authentication method.
- **Auto-enrollment**: After successful KBA, the model is updated automatically.

## Features
- **Real-time typing capture**: Records keystrokes during login attempts.
- **Keystroke-based features**: Features such as **dwell time**, **key press cadence**, and **pause duration** are captured.
- **Authentication model**: The system trains an **SGDClassifier** model to identify users based on their typing dynamics.
- **KBA fallback**: If the model's prediction confidence is low, the system asks a set of security questions to verify the user.
- **Automatic model update**: After successful KBA, the model is updated with the new data.

## Installation

### Prerequisites
1. **Python 3.8+**: Make sure Python is installed on your system.
2. **Install Dependencies**: The project requires a few libraries to run. Use the following command to install them:

   ```bash
   pip install -r requirements.txt
   ```

### Setting Up the Project

1. **Clone this repository** or download the project files.
   
2. **Navigate to your project directory**:

   ```bash
   cd keystroke-auth
   ```

3. **Set up a Python virtual environment**:
   - On Windows:
     ```bash
     python -m venv .venv
     .venv\Scriptsctivate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

1. **Frontend**: 
   - Open `frontend/index.html` in a browser. You can use the [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) extension in Visual Studio Code or any static server.
   - Enter a **username** and type a few sentences in the provided textarea.
   - Click **Enroll sample** to create an enrollment sample.
   - Once enough samples are enrolled (10+), use **Login (score)** to test authentication.

2. **Backend**:
   - **Run the backend server** using **Uvicorn**:
   
     ```bash
     uvicorn backend.app:app --reload
     ```

   - The backend API will run on `http://127.0.0.1:8000`.
   - It provides endpoints for **enrollment**, **login authentication**, and **KBA** (Knowledge-Based Authentication).

3. **Training the Model**:
   - Once you have enough samples (10+), you can train the model by running:
   
     ```bash
     python -m backend.train_util
     ```
   
   - Alternatively, you can trigger model training for a single user by running:
   
     ```bash
     python scripts/train_sgd.py --user <username>
     ```

## Directory Structure

- `backend/`: Contains backend logic, model training, data processing, and security functionality.
- `frontend/`: The UI where users can enroll and log in using their keystroke patterns.
- `scripts/`: Utility scripts for training and testing the models.
- `data/`: Contains raw event data and CSV files for feature extraction and model training.
- `requirements.txt`: A file that lists all the dependencies needed for the project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework.
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools.
- [Orjson](https://github.com/ijl/orjson) for fast JSON serialization.
