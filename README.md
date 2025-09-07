# 🌿 Universal Plant Disease Detection  

A deep learning–based system for detecting plant diseases from images.  
The application uses a **Convolutional Neural Network (CNN)** to classify plant leaves into multiple categories with high accuracy.  

---

## 🚀 Features
- Detects diseases in **vegetables, fruits, cereals, herbs, trees, and tropical plants**  
- Supports **200 plant disease classes**  
- Provides **disease name and confidence score**  
- Simple **web interface using Streamlit**  
- Fast and reliable predictions  

---

## 📦 Installation

1️⃣ Clone the Repository

git clone https://github.com/tajanajain123/Plant-Disease-Detection.git
cd Plant-Disease-Detection

2️⃣ Install Dependencies
Make sure you have Python 3.8+ installed. Then run:
pip install -r requirements.txt

3️⃣ Setup the Model
python setup.py

4️⃣ Start the Application
streamlit run main.py

5️⃣ Open in Browser
Go to 👉 http://localhost:8501

🧠 Model Details
Input: 224x224 RGB images

Architecture: CNN with 5 convolutional blocks and 3 dense layers

Output: 200 plant disease classes

Accuracy: ~90% on known plants

🌱 Supported Plant Classes
Vegetables: Tomato, Potato, Corn, Pepper, Cucumber, Cabbage, etc.

Fruits: Apple, Orange, Grape, Strawberry, Peach, Cherry, etc.

Cereals: Wheat, Rice, Barley, Soybean, etc.

Herbs: Basil, Mint, Rose, etc.

Trees: Oak, Pine, Eucalyptus, etc.

Tropical: Coffee, Tea, Cotton, etc.

🛠️ Requirements
Python: 3.8 or higher

Memory: 4GB+ RAM

Storage: 2GB+ free space

Dependencies: Listed in requirements.txt

🎯 Use Cases
👩‍🌾 Farmers – Identify plant diseases quickly in the field

🔬 Researchers – Reliable analysis for agricultural studies

🎓 Students – Learn about AI in agriculture

🌱 Gardeners – Monitor plant health at home

📊 Example Workflow
Open the web app

Upload a plant leaf image

Get disease prediction with confidence score