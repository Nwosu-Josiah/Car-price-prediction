# 🚗 Car Price Prediction App

An interactive **Streamlit** web app that predicts used car prices based on key attributes such as manufacturer, model, year, mileage, condition, fuel type, and transmission.  

The project covers:
- 🧩 Exploratory Data Analysis (EDA)
- 🧠 Model training and preprocessing
- ⚙️ Interactive prediction dashboard
- ☁️ Streamlit Cloud deployment

---

## 📂 Project Structure

car-price-prediction/
├── app.py 
├── requirements.txt 
├── README.md
│
├── datasets/
│  └── sampled_raw_data.csv 
│ 
├── models/
│ ├── model.json 
│ ├── preprocessor.pkl  
│ └── feature_names.pkl 
│
└── notebooks/
├── Cars_prediction.ipynb # 
└── Exploratory_analysis.ipynb 


---

## ⚙️ Installation & Setup
```bash
1️⃣ Clone the Repository

git clone https://github.com/Nwosu-Josiah/Car-price-prediction
cd Car-price-prediction
2️⃣ Create and Activate a Virtual Environment

python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate  # On macOS/Linux
3️⃣ Install Dependencies

pip install -r requirements.txt
🧠 Reproducibility Workflow
🔍 1. Exploratory Data Analysis
Open notebooks/Exploratory_analysis.ipynb to:

Explore dataset structure

Identify missing values and correlations

Visualize variable distributions

🧩 2. Model Training and Feature Engineering
Open notebooks/Cars_prediction.ipynb to:

Perform data cleaning and feature engineering (e.g., car age)

Train an XGBoost Regressor

Save:

model.json — trained model

preprocessor.pkl — preprocessing pipeline

feature_names.pkl — feature name mappings

🧮 3. Model Deployment with Streamlit
Run the Streamlit app locally with:


python -m streamlit run app.py
Then open the displayed local URL (usually http://localhost:8501) to access the dashboard.

🚀 Deployment on Streamlit Cloud
To deploy the app live:

Push to GitHub

Make sure all required files (app.py, requirements.txt, datasets/, models/) are committed.

Go to Streamlit Cloud

Visit https://share.streamlit.io.

Deploy the App

Click New app

Choose your repository

Set:

Branch: main

Main file path: app.py

Click Deploy 🚀

Your app will be live on a public URL like:


https://your-username-car-price-prediction.streamlit.app
📊 App Features
✅ User-friendly interface built with Streamlit
✅ Dynamic filtering of car models based on manufacturer
✅ Real-time price prediction using a trained XGBoost model
✅ Clean design with dropdowns and numeric inputs
✅ Ready for deployment on Streamlit Cloud

📦 Requirements
All dependencies required to run the project are listed in requirements.txt.


Install them with:


pip install -r requirements.txt
🧱 Local Development Tips
If you moved your files into subfolders (recommended structure):

Update the file paths in app.py like this:

df_raw = pd.read_csv("datasets/sampled_raw_data.csv", low_memory=False)

model.load_model("models/model.json")
preprocessor = joblib.load("models/preprocessor.pkl")
feature_names = list(joblib.load("models/feature_names.pkl"))
This ensures Streamlit Cloud and local runs both locate the right files.

👨‍💻 Author
Developed with ❤️ by Nwosu Josiah
Machine Learning Engineer | Data Scientist

📫 Contact: mwosujosiah20@gmail.com
🌐 GitHub: https://github.com/Nwosu-Josiah

📜 License
This project is open-source under the MIT License.

