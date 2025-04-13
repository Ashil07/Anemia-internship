# Anemia Prediction
## 1. Overview
This project implements a web application for predicting anemia risk using machine learning. The app allows users to input their hematological parameters and receive an anemia diagnosis based on a trained predictive model. It leverages Streamlit for an interactive interface and uses advanced machine learning techniques to assess anemia status.

Traditional diagnostic methods often rely on manual interpretation of hematological parameters, which can be time-consuming and prone to human error. Our research aims to bridge this gap by developing a sophisticated machine learning approach that:

- Provides rapid, data-driven anemia risk assessment
- Uncovers subtle, non-linear relationships in hematological data
- Offers personalized insights based on comprehensive statistical analysis
- Democratizes access to advanced diagnostic tools

Built upon groundbreaking research by [Mojumdar et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11699093/), this project represents a convergence of clinical expertise and cutting-edge data science, transforming how we approach anemia diagnosis and early intervention.

Check out the live Streamlit App [here](https://anemia-diagnosis.streamlit.app/).

## 2. Features

- **Exploratory Data Analysis**: Comprehensive analysis of anemia dataset
- **Machine Learning Models**: Training and evaluation of multiple classification models
- **Streamlit Web Application**: User-friendly interface for anemia prediction
- **Clinical Interpretation**: Medically-relevant insights about prediction results
- **Visualization**: Interactive charts and graphs for data exploration

## 3. Project Structure

```
anemia_prediction/
├── config/
│   └── feature_config.py     # Feature configuration and normal ranges
├── data/
│   └── Anemia Dataset.xlsx   # Dataset from Aalok Healthcare Ltd.
├── models/
│   └── anemia_prediction_model.pkl  # Saved trained model
├── output/
│   ├── figures/              # EDA visualizations
│   └── model_evaluation/     # Model evaluation results
├── src/
│   ├── data_exploration.py   # EDA script
│   ├── model_training.py     # Model training script
│   ├── utils/
│   │   ├── preprocessing.py  # Data preprocessing utilities
│   │   └── evaluation.py     # Model evaluation utilities
│   └── visualization.py      # Visualization functions
├── app.py                    # Streamlit application entry point
└── requirements.txt          # Dependencies
```

## 4. Installation

1. Clone the repository:
```bash
git clone https://github.com/kimnguyen2002/Anemia-Prediction.git
cd Anemia-Prediction
```

2. **Install the required packages:**
Create a virtual environment (optional but recommended) and install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

4. Open your web browser and navigate to `http://yourlocalhost` to access the application.

## 5. Usage

### Data Exploration

To run the exploratory data analysis:

```bash
python src/data_exploration.py
```

This will generate visualizations and statistics about the dataset in the `output/figures` directory.

### Model Training

To train and evaluate machine learning models:

```bash
python src/model_training.py
```

This will:
- Load and preprocess the dataset
- Perform feature selection
- Train multiple machine learning models
- Evaluate their performance
- Save the best model to `models/anemia_prediction_model.pkl`

### Web Application

To run the Streamlit web application:

```bash
streamlit run app.py
```

This will start the application on your local machine, which you can access via your web browser.

## 6. Web Application Features
1. **About**: Information about anemia and the research background
2. **Prediction Tool**: Enter patient data and get anemia prediction with confidence level
3. **Exploratory Analysis**: Interactive visualizations of the dataset
4. **Interpretation Guide**: Medical interpretation of hematological parameters

## 7. Dataset
Data was collected from anemia patients at Aalok Healthcare Ltd., located in Dhaka, Bangladesh on 9th October 2023 (https://data.mendeley.com/datasets/y7v7ff3wpj/1)
The dataset includes the following hematological parameters:
- Gender (Male/Female)
- Age
- Hemoglobin Level (Hb)
- Red Blood Cell Count (RBC)
- Packed Cell Volume (PCV)
- Mean Corpuscular Volume (MCV)
- Mean Corpuscular Hemoglobin (MCH)
- Mean Corpuscular Hemoglobin Concentration (MCHC)

## 8. Collaboration
A special thank you to Dr. Gem Wu from Chang Gung Memorial Hospital for insightful medical guidance, and to my dedicated data science collaborator who contributed significantly to this project's development. This project is the result of a unique interdisciplinary collaboration:

### Clinical Expertise
[Dr. Gem Wu](https://scholar.google.com.tw/citations?user=MwIr5fMAAAAJ&hl=en) | Hematologist, Chang Gung Memorial Hospital

- Github: [Dr. Gem Wu's Github](https://github.com/Gem-Wu)
- Provided critical medical insights
- Checked the medical accuracy of the diagnostic approach
### Data Science
Ngoc Duy | Data Scientist

- Github: [Ngoc Duy's Github](https://github.com/NgocDuy3112)
- Developed advanced machine learning models
- Implemented complex data analysis techniques

## 9. References
- Paper: Mojumdar et al., "AnaDetect: An extensive dataset for advancing anemia detection, diagnostic methods, and predictive analytics in healthcare", PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC11699093/)
- Anemia: Approach and Evaluation (https://manualofmedicine.com/topics/hematology-oncology/anemia-approach-and-evaluation/)
- Source Code (Anemia Detection with Machine Learning): "Anemia Detection with Machine Learning", GitHub repository (https://github.com/maladeep/anemia-detection-with-machine-learning)
- Source Code (Anemia Prediction): "Anemia Prediction", GitHub repository (https://github.com/muscak/anemia-prediction)

## 10. Disclaimer
This application is for educational and research purposes only and should not replace professional medical advice.

## 11. License

This project is licensed under the MIT License.
