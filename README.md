# 🩺 Smart Health Monitoring System (SHMS)

## 🌟 Project Overview

The Smart Health Monitoring System (SHMS) is a Python-based web application designed to help users proactively monitor their well-being. By allowing manual entry of daily health metrics (Weight, BP, Sugar, Sleep, Stress), the system leverages Machine Learning models to provide early risk predictions, trend visualizations, and personalized lifestyle recommendations.

This project was built using the **Flask** micro-framework, making it a scalable and maintainable solution for data-driven applications.

## 🚀 Key Features

| Module | Functional Requirements | Status |
| :--- | :--- | :--- |
| **1. User Authentication (R6)** | Secure Email-based Login/Signup/Logout. | Complete |
| **2. Health Data Management (R1)** | CRUD operations (Create, View, Edit, Delete) for daily health records. | Complete |
| **3. AI Risk Analysis (R3, R4)** | Displays real-time risk assessment (High/Medium/Low) based on data rules and integrated ML models (Diabetes/Heart Disease). | Complete |
| **4. Data Visualization (R5)** | Interactive trend graphs (Daily/Weekly/Monthly) for key metrics (Weight, BP, Sugar). | Complete |
| **5. Professional Reporting (R5.4)**| Generation of detailed PDF Health Reports (using ReportLab) summarizing risk and historical data. | Complete |
| **6. Expert UI/UX** | Professional, high-contrast, and minimalist user interface design. | Complete |

## 🛠️ Technology Stack

* **Backend Framework:** Flask
* **Language:** Python 3.x
* **Database:** SQLite (SQLAlchemy ORM)
* **Machine Learning:** Scikit-learn, NumPy, Pandas
* **Data Visualization:** Matplotlib
* **Reporting:** ReportLab
* **Frontend:** HTML5, CSS3 (Custom Styling), Bootstrap 5

## 💻 Installation and Setup

Follow these steps to set up and run the project locally.

### Prerequisites

You must have Python 3.9+ installed.

1.  **Clone the Repository (If not already done):**

    ```bash
    git clone [https://github.com/APNA_USERNAME/Smart-Health-Monitor.git](https://github.com/APNA_USERNAME/Smart-Health-Monitor.git)
    cd Smart-Health-Monitor
    ```

2.  **Create and Activate Virtual Environment:**

    ```bash
    python -m venv venv
    # For Windows:
    .\venv\Scripts\activate
    # For macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Required Libraries:**

    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file first using `pip freeze > requirements.txt`)*

### Database and ML Model Preparation

The application requires a database and pre-trained ML models to function.

1.  **Initialize Database (Creates `health_db.db`):**

    ```bash
    python database.py
    ```

2.  **Train and Save ML Models (Creates `model_diabetes.pkl` and `model_heart.pkl`):**
    *(Ensure you have downloaded the datasets into the `datasets/` folder first.)*

    ```bash
    python ml_models.py
    ```

### Running the Application

1.  **Start the Flask Server:**

    ```bash
    python app.py
    ```

2.  **Access the App:** Open your web browser and navigate to:
    `http://127.0.0.1:5000/`

## ⚙️ Usage Guide

1.  **Sign Up:** Register a new account using your **Email Address** (the primary login credential).
2.  **Login:** Sign in with your registered email and password.
3.  **Data Entry (R1):** Navigate to the **"1. Health Data Entry"** section to add new daily records.
4.  **Risk Check (R3):** Your **Overall Risk Level** and personalized recommendations will automatically update on the dashboard based on your latest entry.
5.  **View Trends (R5):** Click **"View Trends & Graphs"** to see dynamic charts of your historical health metrics.
6.  **Download Report (R5.4):** Click **"Generate PDF Report"** to download a detailed professional summary of your risk profile and data tables.

---

## 🤝 Contribution and Support

For any issues or suggestions, please open an issue in this repository.

***
*Developed for University Project (Final Submission)*
***
