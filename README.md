# CO2 Emissions from Vehicles Marketed in France

This project analyzes the **ADEME Car Labelling dataset** to study the factors influencing **vehicle CO₂ emissions** in France. The workflow includes **data cleaning, preprocessing, exploratory data analysis (EDA), clustering, and predictive modeling**.

The analysis explores how **vehicle mass, engine power, fuel type, gearbox, and body type** relate to CO₂ emissions. **Clustering techniques** are used to identify distinct vehicle segments, while **machine learning models** are developed to predict CO₂ emissions and determine the most influential vehicle characteristics.

# 1. Dataset Description

The dataset used in this analysis comes from the French open data portal **data.gouv.fr**.

- **Dataset:** CO₂ and pollutant emissions of vehicles sold in France  
- **Provider:** ADEME (Agence de la transition écologique)

🔗 [View the dataset on data.gouv.fr](https://www.data.gouv.fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france?resource_id=6ff09b59-84ca-4346-a8d1-3587ed94da15)

It contains **44,850 vehicle records** and includes information on a wide range of **vehicle specifications and environmental indicators**.

After translating the original **French column names into English**, the main variables used in the analysis include:

### Vehicle Identification
- `Brand` – Manufacturer of the vehicle  
- `Folder Model` – Vehicle model grouping  
- `Body` – Vehicle body type  

### Powertrain and Technical Characteristics
- `Fuel` – Fuel or energy type  
- `Hybrid` – Indicates whether the vehicle is hybrid  
- `Gearbox` – Transmission type  
- `Administrative Power` – Fiscal horsepower  
- `Maximum Power (kW)` – Engine power in kilowatts  

### Fuel Consumption and Emissions
- `Combined Consumption (l/100km)` – Fuel consumption  
- `CO2 (g/km)` – Carbon dioxide emissions  

### Vehicle Mass
- `Empty Mass Euro Min (kg)` – Minimum empty mass  
- `Empty Mass Euro Max (kg)` – Maximum empty mass  

A new variable was created:

- `Empty Mass Euro Avg (kg)` – Average vehicle mass calculated from the minimum and maximum empty mass values.

# 2. Technology Stack

The analysis was conducted in Python using the following tools and libraries:  

* **Python:** Core language for analysis  
* **Pandas:** Data loading, cleaning, transformation, and summarization  
* **NumPy:** Numerical operations  
* **Matplotlib and Seaborn:** Data visualization  
* **Scikit-learn:** Clustering and predictive modeling  
* **Google Colab:** Execution environment with Google Drive integration  

# 3. Exploratory Data Analysis (EDA)

The exploratory analysis examined the structure and patterns of the dataset from several perspectives.  

## 3.1 Descriptive Overview

The dataset contains:

- **44,850** rows  
- **51** brands  
- **458** folder models  
- **13** fuel categories  

#### Average Values

- `Administrative Power`: **11.02**  
- `Maximum Power (kW)`: **124.78 kW**  
- `Combined Consumption (l/100km)`: **7.71 l/100km**  
- `CO2 (g/km)`: **198.74 g/km**  
- `Empty Mass Euro Avg (kg)`: **2120.25 kg**

## 3.2 Distribution of Brands and Models

### Brand Distribution in the Dataset

The dataset contains **51 car brands**, but the distribution is **highly uneven**.

- **Mercedes-Benz** has by far the **largest number of observations (~38,000)**, mainly driven by frequently registered models such as **Viano, Vito, and Sprinter**.

### Model Diversity by Brand

- **Audi** has the **highest number of distinct models**.
- It is followed by **Volkswagen** and **Porsche**.
- **Mercedes-Benz** also shows a **relatively broad model range**.

**Key insight:**  
While **Mercedes-Benz dominates the dataset in volume**, **Audi leads in model diversity**.

<img width="1638" height="784" alt="Screenshot 2026-03-10 at 2 24 47 PM" src="https://github.com/user-attachments/assets/1a86cf2c-b703-49db-a182-bc1b2246ba47" />

<img width="1619" height="776" alt="Screenshot 2026-03-10 at 5 37 44 PM" src="https://github.com/user-attachments/assets/ff978cd2-113e-4cf3-8f56-a7e5735a0b37" />


## 3.3 Distribution Fuel and Body Type 

- **Minibuses dominate the dataset**, with far more entries than any other body type.  
- Other body types such as **berlines (sedans), breaks (wagons), and SUVs** appear but at much lower frequencies.  
- Smaller segments like **coupes, cabriolets, and compact vehicles** are relatively rare.
- In terms of fuel, **diesel (`GO`) vehicles overwhelmingly dominate the dataset**, followed by **gasoline (`ES`) vehicles**.  
- **Hybrid, electric, and alternative fuels** appear only in **very small numbers**, reflecting the market structure at the time of data collection.

**Key insight:**  
The dataset is strongly skewed toward **diesel-powered minibuses and conventional combustion vehicles**.
<img width="1584" height="784" alt="body fuel" src="https://github.com/user-attachments/assets/d5b78398-eef8-4bc2-bf63-bca01587cfed" />


# 4 Reducing Brand Imbalance through Deduplication

To avoid bias from repeatedly occurring vehicle configurations, the dataset was reduced to **unique vehicle specifications**. Many vehicles appear multiple times in the original dataset because the same technical configuration can be listed across different entries. Keeping all of them would overrepresent certain brands or models.

To address this, duplicates were removed based on the following variables that define a vehicle’s technical specification:

- `Brand`
- `Folder Model`
- `Fuel`
- `Body`
- `Gearbox`
- `Maximum Power (kW)`
- `Empty Mass Euro Avg (kg)`
- `Combined Consumption (l/100km)`
- `CO2 (g/km)`

Two observations are considered identical if all of these variables have the same values. In such cases, only one observation is retained.
This deduplication step reduces **brand imbalance** and ensures that the subsequent analysis focuses on **distinct vehicle configurations** rather than repeated instances of the same specification.

<img width="791" height="410" alt="Screenshot 2026-03-16 at 3 12 58 PM" src="https://github.com/user-attachments/assets/9bd72a07-e5a6-4d04-8f6a-d9cc9e8777ab" />


# 5 Power and Weight Effects on CO₂ Emissions (after Deduplication)

The plots examine how `Empty Mass Euro Avg (kg)` and `Maximum Power (kW)` relate to `CO2 (g/km)`.

## 5.1 Relationship Empty Mass (kg) vs. CO₂ (g/km)

- There is a **strong positive correlation** between `Empty Mass Euro Avg` and `CO₂ (g/km)` *(r = 0.6817, R² = 0.4648)*.
- **Heavier vehicles generally produce higher CO₂ emissions**, as indicated by the upward regression trend.
- The **highest observation density** occurs around **1400–1800 kg** for `empty_mass_euro_avg` and **120–200 g/km** for `CO2 (g/km)`.
- The distribution of `Empty Mass Euro Avg` peaks around **1500–1800 kg**, with relatively fewer vehicles above **2500 kg**.
- The distribution of `CO2 (g/km)` is **right-skewed**, with most vehicles between **120–220 g/km** and a few high-emission outliers.
- Overall, ``Empty Mass Euro Avg`` is a **significant but not exclusive predictor** of `CO2 (g/km)`.
<img width="1584" height="783" alt="Mass vs  CO2" src="https://github.com/user-attachments/assets/c74361e1-f862-4af4-a5d2-928c952ac5aa" />

### 5.2 Relationship Maximum Power (kW) vs. CO2 (g/km)

- There is a **moderate positive correlation** between `maximum_power (kw)` and `CO₂ (g/km)` *(r = 0.6703, R² = 0.4493)*.
- Vehicles with **higher engine power generally produce higher CO₂ emissions**, as shown by the upward regression trend.
- The **highest observation density** occurs around **80–130 kW** for `Maximum Power (kW)` and **110–180 g/km** for `CO2 (g/km)`.
- The distribution of `Maximum Power (kW)` is **right-skewed**, with most vehicles clustered between **70–140 kW** and fewer high-power vehicles above **250 kW**.
- The distribution of `CO2 (g/km)` is also **right-skewed**, with most vehicles between **120–220 g/km** and a small number of high-emission outliers.
- Overall, `maximum_power (kw)` is a **relevant but not sole predictor** of `CO2 (g/km)`.

<img width="1584" height="783" alt="Power vs  CO2" src="https://github.com/user-attachments/assets/a5d045f6-f934-461f-ad3f-ac65367fa806" />


## 6. Clustering Analysis Results

### 6.1  Clustering Dashboard 

The dashboard summarizes the **size, power, mass, and emission differences across clusters**.

- **Cluster size:** `cluster 0` is the **largest**, followed by `cluster 1` and `cluster 2`, while `cluster 3` is the **smallest** segment.
- **CO₂ distribution:**  
  `cluster 2` shows the **lowest `co2 (g/km)`**, `cluster 0` **moderate**, `cluster 1` **higher**, and `cluster 3` the **highest emissions with greater variability**.
- **Power–mass relationship:**  
  Clusters are mainly separated by **vehicle size and performance**:
  - `cluster 2` → **efficiency vehicles**
  - `cluster 0` → **mid-range vehicles**
  - `cluster 1` → **utility vehicles**
  - `cluster 3` → **performance vehicles**
- **Mean emissions vs fleet average (~171 g/km):**  
  `cluster 2` and `cluster 0` are **below average**, while `cluster 1` and `cluster 3` are **above average**.
<img width="1584" height="784" alt="Clustering Dashboard" src="https://github.com/user-attachments/assets/031042d3-7586-4153-a786-00d2466d54dd" />
<img width="1780" height="584" alt="Categorical Features Distribution" src="https://github.com/user-attachments/assets/df4cd8db-7de7-4553-bbc2-1f40d27202a3" />
---

### 6.2 Cluster Profiles (Radar Chart & Heatmap)

The radar chart and heatmap show the **normalized cluster** for  
`Maximum Power (kW)`, `Empty Mass Euro Avg (kg)`, and `CO2 (g/km)`.


<img width="1552" height="818" alt="Vehicle Cluster Profiles" src="https://github.com/user-attachments/assets/cd9709c4-1c90-4faa-b60a-dccfa6e04e80" />

### 6.3 Key Insight

The clustering identifies **four distinct vehicle segments**, primarily structured by  
`Maximum power (kw)`, `Empty Mass Euro Avg (kg)`, and resulting `CO2 (g/km)` emissions.  
As **vehicle power and mass increase, average emissions increase as well**, producing clearly differentiated emission profiles across clusters.


# 5. Predictive Modeling and Interpretation

A regression pipeline was built to **predict `CO2 (g/km)`** using the features  
`Body`, `Fuel`,`Gearbox`, `Maximum Power (kW)` and `Empty Mass Euro Avg (kg)`

- First, the modeling dataset was prepared from `df_unique`, missing values were removed, and the data was split into **training and test sets**.
- Two **preprocessing strategies** were applied:
  - **One-hot encoding + scaling** for linear models
  - **One-hot encoding with numeric passthrough** for tree-based models
- Five regression models were trained and compared:
  - `Linear Regression`
  - `Ridge`
  - `Lasso`
  - `Random Forest`
  - `Gradient Boosting`

Model performance was evaluated using **`R²` and `MAE`** on both training and test data to assess **predictive accuracy and potential overfitting**.

<img width="1583" height="919" alt="Model Performance" src="https://github.com/user-attachments/assets/0ee5e16f-4a05-46e7-afc6-5d5353c9b83d" />


After comparing models, the **Random Forest** model was further analyzed:

- **Feature importance** identifies the most influential predictors of `CO2 (g/km)`
<img width="1584" height="884" alt="Feature Importance" src="https://github.com/user-attachments/assets/74f6e159-8fb4-4399-aaf2-d8929dd3c25c" />

- **Partial Dependence Plots (PDPs)** show how `Maximum Power (kW)`, `Empty Mass Euro Avg (kg)`, `Fuel`, `Body`, and `Gearbox` influence predicted emissions
<img width="1584" height="919" alt="Partial Dependency Plots" src="https://github.com/user-attachments/assets/b34c0d5c-5c26-4af5-8756-0764996ec55b" />

- A **SHAP summary plot** explains the contribution and direction of each feature across observations
<img width="1474" height="884" alt="Shap plot" src="https://github.com/user-attachments/assets/8e969318-d129-4fff-8a9c-b3e2a7af3559" />

- Finally, a **single decision tree** from the Random Forest was visualized to illustrate how predictions are generated
<img width="1556" height="884" alt="Decision Tree" src="https://github.com/user-attachments/assets/b420908f-81bb-48dd-98e6-33c9da49caa1" />

Overall, this step combines **model comparison and explainability** to better understand how vehicle characteristics influence `CO2 (g/km)` emissions.

