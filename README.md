# CO2 and pollutant emissions from vehicles sold in France
This README provides an overview of the data analysis conducted on the Car Labelling dataset from ADEME France. The analysis focuses on understanding vehicle characteristics, cleaning and preprocessing the dataset, performing exploratory data analysis (EDA), identifying patterns through clustering, and building predictive models for CO2 emissions. The study highlights how fuel type, vehicle mass, engine power, gearbox, and body type relate to fuel consumption and environmental impact.

# 1. Dataset Description

The dataset was loaded from the file:

`cl_JUIN_2013-complet3.csv`

It contains **44,850 vehicle records** and includes information on a wide range of **vehicle specifications and environmental indicators**.

After translating the original **French column names into English**, the main variables used in the analysis include:

### Vehicle Identification
- `Brand` – Manufacturer of the vehicle  
- `Folder Model` – Vehicle model grouping  
- `Range` – Vehicle segment  
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
- `CO type 1 (g/km)` – Carbon monoxide emissions  
- `HC (g/km)` – Hydrocarbons  
- `NOX (g/km)` – Nitrogen oxides  
- `HC+NOX (g/km)` – Combined hydrocarbons and nitrogen oxides  
- `Particles (g/km)` – Particulate emissions  

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



# 3. Data Preprocessing and Cleaning

Several preprocessing and data-cleaning steps were performed before the analysis.

### 3.1 Data Import and Column Standardization

- The CSV file was loaded using **`latin1` encoding** and **semicolon (`;`) separation**.
- Original **French column names** were translated into **English** to improve readability and usability.

### 3.2 Missing Value Analysis

A missing-value audit showed that several **pollutant variables** had substantial missing values, particularly:

- `HC (g/km)`
- `HC+NOX (g/km)`
- `Particles (g/km)`
- `CO type 1 (g/km)`
- `NOX (g/km)`

In contrast, **fuel consumption and CO₂ variables** contained only a **small number of missing observations**.

<img width="1729" height="580" alt="Screenshot 2026-03-09 at 6 16 07 PM" src="https://github.com/user-attachments/assets/98df60fb-6d5e-4ccf-8836-6a10b50bac2c" />

### 3.3 Pollutant Reconstruction

Missing pollutant values were partially reconstructed using the relationship:

\[
HC + NOX = HC + NOX
\]

This allowed the calculation of missing **HC** and **NOX** values where possible, improving the **completeness of the dataset**.

### 3.4 Correction of Inconsistent Categorical Entries

Some erroneous **gearbox** and **regulatory field (Field V9)** values were corrected.

**Gearbox corrections:**

- `N 0`, `N 1` → `A 0`
- `S 6` → `D 6`

### 3.5 Electric Vehicle Adjustments

For **electric vehicles (`Fuel == "EL"`)**, several **emissions and fuel-consumption variables** were replaced with `0`, since these vehicles do not produce **tailpipe CO₂ or combustion-related pollutants**.

### 3.6 Mass Aggregation

The variables:

- `Empty Mass Euro Min (kg)`
- `Empty Mass Euro Max (kg)`

were merged into a single variable:

- **`Empty Mass Euro Avg (kg)`**

This **average vehicle mass** was then used throughout the analysis.


# 4. Exploratory Data Analysis (EDA)

The exploratory analysis examined the structure and patterns of the dataset from several perspectives.  

## 4.1 Descriptive Overview

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


|index|Brand|Folder Model|Utac Model|Commerical Designation|cnit|Type Variant Version|Fuel|Hybrid|Administrative Power|Maximum Power \(kW\)|Gearbox|Urban Consumption \(l/100km\)|Extra Urban Consumption \(l/100km\)|Combined Consumption \(l/100km\)|CO2 \(g/km\)|CO type 1 \(g/km\)|HC \(g/km\)|NOX \(g/km\)|HC+NOX \(g/km\)|Particles \(g/km\)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|44850|44850|44850|44850|44850|44850|44850|44850|44850\.0|44850\.0|44850|44847\.0|44847\.0|44850\.0|44850\.0|44586\.0|44586\.0|44547\.0|44586\.0|41747\.0|
|unique|51|458|419|3582|44191|28781|13|2|NaN|NaN|13|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|top|MERCEDES-BENZ|VIANO|VIANO|VIANO 2\.2 CDI|M10LNCVP000R207|263AXG1B05|GO|non|NaN|NaN|M 6|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|freq|38450|14031|14031|5874|16|32|37778|44593|NaN|NaN|19364|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|mean|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|11\.018996655518395|124\.78083389074695|NaN|9\.698303119495174|6\.561922759604879|7\.709544035674469|198\.7379264214047|0\.1533268088637689|0\.022389081774548063|0\.31183959413652995|0\.33395590544116993|0\.000959843581574724|
|std|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|5\.554474663340535|49\.15880390757593|NaN|2\.3823957247243936|1\.2112840762678807|1\.6125365011865005|39\.43598817742501|0\.13899707813825268|0\.014255190181952957|0\.4631107921136125|0\.4582791644245911|0\.006465942752424883|
|min|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|1\.0|10\.0|NaN|0\.0|0\.0|0\.0|0\.0|0\.0|-0\.05399999999999999|0\.001|0\.0|0\.0|
|25%|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|9\.0|100\.0|NaN|8\.8|6\.3|7\.2|187\.0|0\.046|0\.01100000000000001|0\.158|0\.18|0\.0|
|50%|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|10\.0|120\.0|NaN|9\.7|6\.7|7\.7|203\.0|0\.093|0\.01899999999999999|0\.197|0\.216|0\.001|
|75%|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|11\.0|125\.0|NaN|10\.7|7\.1|8\.4|221\.0|0\.222|0\.03|0\.228|0\.253|0\.001|
|max|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|81\.0|559\.3|NaN|41\.1|14\.9|24\.5|572\.0|0\.968|0\.143|1\.846|1\.86|0\.61|


## 4.2 Distribution of Energy Types

The dataset is dominated by **diesel (`GO`) vehicles**, followed by **gasoline (`ES`) vehicles**.

Other categories, such as **electric** and **hybrid vehicles**, are present but **much less frequent**, reflecting the **market structure at the time of data collection**.

<img width="1387" height="570" alt="Screenshot 2026-03-10 at 5 03 49 PM" src="https://github.com/user-attachments/assets/3199f892-ede8-43ee-9ad4-3fc02993c078" />

**Terminology for energy types**

<table>
<tr>
<td width="50%">

<small>
<ul>
<li><b>ES</b>: unleaded 95 gasoline vehicles</li>
<li><b>GO</b>: diesel vehicles</li>
<li><b>FE</b>: E85 vehicles</li>
<li><b>EH</b>: gasoline (ES) / electric hybrid (non-rechargeable)</li>
<li><b>EE</b>: gasoline (ES) / electric hybrid (rechargeable)</li>
<li><b>GH</b>: diesel / electric hybrid (non-rechargeable)</li>
<li><b>GL</b>: diesel / electric hybrid (rechargeable)</li>
<li><b>EL</b>: electric</li>
</ul>
</small>

</td>
<td width="50%">

<small>
<ul>
<li><b>ES / GP</b>: dual fuel gasoline (ES) / liquefied petroleum gas (GPL) (gasoline consumption data)</li>
<li><b>GP / ES</b>: dual fuel gasoline (ES) / liquefied petroleum gas (GPL) (GPL consumption data)</li>
<li><b>ES / GN</b>: natural gas (GNV) (gasoline consumption data)</li>
<li><b>GN / ES</b>: dual fuel gasoline (ES) / natural gas for vehicles (GNV) (GNV consumption)</li>
<li><b>GN</b>: single fuel natural gas for vehicles (GNV consumption data)</li>
</ul>
</small>

</td>
</tr>
</table>

## 4.3 Distribution of Brands and Models

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


### Reducing Brand Imbalance through Deduplication

To avoid bias from repeatedly occurring vehicle configurations, the dataset was reduced to **unique non-electric vehicle specifications**. Many vehicles appear multiple times in the original dataset because the same technical configuration can be listed across different entries. Keeping all of them would overrepresent certain brands or models.

To address this, duplicates were removed based on the following variables that define a vehicle’s technical specification:

- `Brand`
- `Fuel`
- `Body`
- `Gearbox`
- `Maximum Power (kW)`
- `Empty Mass Euro Avg (kg)`
- `CO2 (g/km)`

Two observations are considered identical if all of these variables have the same values. In such cases, only one observation is retained.
This deduplication step reduces **brand imbalance** and ensures that the subsequent analysis focuses on **distinct vehicle configurations** rather than repeated instances of the same specification.

<img width="791" height="410" alt="Screenshot 2026-03-16 at 3 12 58 PM" src="https://github.com/user-attachments/assets/9bd72a07-e5a6-4d04-8f6a-d9cc9e8777ab" />

## 4.4 Distribution of Key Vehicle Characteristics

The histograms illustrate the distributions of several key vehicle characteristics:

- Most vehicles have `CO2 (g/km)` values around **180–230 g/km**.
- `Empty Mass Euro Avg (kg)` is typically between **1900–2200 kg**.
- `Maximum Power (kW)` mostly ranges from **80–140 kW**.
- `Combined Consumption (l/100km)` is concentrated around **6–9 l/100 km**.

<img width="1609" height="794" alt="Screenshot 2026-03-10 at 7 40 56 PM" src="https://github.com/user-attachments/assets/7e5a81a2-e3b1-4ad6-842e-43a99074c0ff" />


## 4.5 Brand-related CO₂ Emissions 

### Distribution CO₂ Emissions by Brand

There is **substantial variation in CO₂ emissions across brands**.

- **Luxury and performance brands** (e.g., *Bentley, Lamborghini, Aston Martin*) generally show **higher median CO₂ emissions**.
- **Mass-market brands** (e.g., *Toyota, Peugeot, Renault*) tend to have **lower medians and more moderate ranges**.
- Some brands display **wide distributions**, indicating that they produce both **low-emission and high-emission vehicles**.

<img width="1609" height="779" alt="Screenshot 2026-03-10 at 5 55 31 PM" src="https://github.com/user-attachments/assets/e0a14cc8-8ab4-4f0f-b464-71542025a7c1" />

### Brand-Level Averages of Vehicle Performance and Emissions ###

The figure presents **brand-level averages of vehicle characteristics** using two **bubble scatterplots**

#### Key Observations

- Both plots show a **positive relationship**:  
  brands with **higher engine power** and **heavier vehicles** tend to have **higher CO₂ emissions**.
- **Luxury and performance brands** (e.g., *Lamborghini, Bentley, Rolls-Royce, Maybach*) appear in the **upper-right region**, reflecting **high power/mass and high emissions**.
- **Smaller or efficiency-focused brands** (e.g., *Smart*) show **lower power, lower mass, and lower emissions**.
- **Electric manufacturers** such as *Tesla* appear **near zero CO₂ emissions**.

**Key insight:**  
Vehicle **power and weight are strongly associated with higher CO₂ emissions across brands**.

<img width="1609" height="797" alt="Screenshot 2026-03-10 at 7 25 37 PM" src="https://github.com/user-attachments/assets/970c2f4f-0399-4f8b-be26-528c70de7153" />

## 4.5 Power and Weight Effects on Emissions and Fuel Consumption

The plots examine how `Empty Mass Euro Avg (kg)` and `Maximum Power (kW)` relate to `CO2 (g/km)`.

### Relationship Empty Mass (kg) vs. CO₂ (g/km)

- There is a **moderate positive correlation** between `empty_mass_euro_avg` and `CO₂ (g/km)` *(r = 0.6817, R² = 0.4648)*.
- **Heavier vehicles generally produce higher CO₂ emissions**, as indicated by the upward regression trend.
- The **highest observation density** occurs around **1400–1800 kg** for `empty_mass_euro_avg` and **120–200 g/km** for `CO2 (g/km)`.
- The distribution of `empty_mass_euro_avg` peaks around **1500–1800 kg**, with relatively fewer vehicles above **2500 kg**.
- The distribution of `CO2 (g/km)` is **right-skewed**, with most vehicles between **120–220 g/km** and a few high-emission outliers.
- Overall, `empty_mass_euro_avg` is a **significant but not exclusive predictor** of `CO2 (g/km)`.
<img width="1584" height="783" alt="Mass vs  CO2" src="https://github.com/user-attachments/assets/c74361e1-f862-4af4-a5d2-928c952ac5aa" />

### Relationship Maximum Power (kW) vs. CO2 (g/km)

- There is a **moderate positive correlation** between `maximum_power (kw)` and `CO₂ (g/km)` *(r = 0.6703, R² = 0.4493)*.
- Vehicles with **higher engine power generally produce higher CO₂ emissions**, as shown by the upward regression trend.
- The **highest observation density** occurs around **80–130 kW** for `maximum_power (kw)` and **110–180 g/km** for `co2 (g/km)`.
- The distribution of `Maximum Power (kW)` is **right-skewed**, with most vehicles clustered between **70–140 kW** and fewer high-power vehicles above **250 kW**.
- The distribution of `CO2 (g/km)` is also **right-skewed**, with most vehicles between **120–220 g/km** and a small number of high-emission outliers.
- Overall, `maximum_power (kw)` is a **relevant but not sole predictor** of `CO2 (g/km)`.

<img width="1584" height="783" alt="Power vs  CO2" src="https://github.com/user-attachments/assets/a5d045f6-f934-461f-ad3f-ac65367fa806" />



## Clustering Analysis Results

### Clustering Dashboard (2×2 Overview)

The dashboard summarizes the **size, power, mass, and emission differences across clusters**.

- **Cluster size:** `cluster 0` is the **largest**, followed by `cluster 1` and `cluster 2`, while `cluster 3` is the **smallest** segment.
- **CO₂ distribution:**  
  `cluster 2` shows the **lowest `co2 (g/km)`**, `cluster 0` **moderate**, `cluster 1` **higher**, and `cluster 3` the **highest emissions with greater variability**.
- **Power–mass relationship:**  
  Clusters are mainly separated by **vehicle size and performance**:
  - `cluster 2` → **low `maximum_power (kw)` and low `empty_mass_euro_avg (kg)`**
  - `cluster 0` → **mid-range vehicles**
  - `cluster 1` → **utility vehicles**
  - `cluster 3` → **high-power, heavier vehicles**
- **Mean emissions vs fleet average (~171 g/km):**  
  `cluster 2` and `cluster 0` are **below average**, while `cluster 1` and `cluster 3` are **above average**.
<img width="1584" height="784" alt="Clustering Dashboard" src="https://github.com/user-attachments/assets/031042d3-7586-4153-a786-00d2466d54dd" />
<img width="1780" height="584" alt="Categorical Features Distribution" src="https://github.com/user-attachments/assets/df4cd8db-7de7-4553-bbc2-1f40d27202a3" />
---



### 2. Cluster Profiles (Radar Chart & Heatmap)

The radar chart and heatmap show the **normalized cluster centroids** for  
`maximum_power (kw)`, `empty_mass_euro_avg (kg)`, and `co2 (g/km)`.


<img width="1552" height="818" alt="Vehicle Cluster Profiles" src="https://github.com/user-attachments/assets/cd9709c4-1c90-4faa-b60a-dccfa6e04e80" />


---



---

### Key Insight

The clustering identifies **four distinct vehicle segments**, primarily structured by  
`maximum_power (kw)`, `empty_mass_euro_avg (kg)`, and resulting `co2 (g/km)` emissions.  
As **vehicle power and mass increase, average emissions increase as well**, producing clearly differentiated emission profiles across clusters.


