# CO2 and pollutant emissions from vehicles sold in France
This README provides an overview of the data analysis conducted on the Car Labelling dataset from ADEME France. The analysis focuses on understanding vehicle characteristics, cleaning and preprocessing the dataset, performing exploratory data analysis (EDA), identifying patterns through clustering, and building predictive models for CO2 emissions. The study highlights how fuel type, vehicle mass, engine power, gearbox, and body type relate to fuel consumption and environmental impact.



# 1. Dataset Description

The dataset was loaded from the file:

cl_JUIN_2013-complet3.csv

It contains 44,850 vehicle records and includes information on a wide range of vehicle specifications and environmental indicators. After renaming the original French columns into English, the main variables used in the analysis included:

* **Brand:** Manufacturer of the vehicle
* **Folder Model:** Vehicle model grouping
* **Fuel:** Fuel or energy type
* **Hybrid:** Indicates whether the vehicle is hybrid
* **Administrative Power:** Fiscal horsepower
* **Maximum Power (kW):** Engine power in kilowatts
* **Gearbox:** Transmission type
* **Combined Consumption (l/100km):** Fuel consumption measure
* **CO2 (g/km):** Carbon dioxide emissions
* **CO type 1 (g/km), HC (g/km), NOX (g/km), HC+NOX (g/km), Particles (g/km):** Pollutant emissions
* **Empty Mass Euro Min / Max (kg):** Minimum and maximum empty mass
* **Body:** Vehicle body type
* **Range:** Vehicle segment

A new feature, Empty Mass Euro Avg (kg), was created by averaging the minimum and maximum empty mass values.



# 2. Technology Stack

The analysis was conducted in Python using the following tools and libraries:  

* **Python:** Core language for analysis  
* **Pandas:** Data loading, cleaning, transformation, and summarization  
* **NumPy:** Numerical operations  
* **Matplotlib and Seaborn:** Data visualization  
* **Scikit-learn:** Clustering and predictive modeling  
* **Google Colab:** Execution environment with Google Drive integration  



# 3. Data Preprocessing

Several preprocessing and cleaning steps were performed before analysis:

**3.1 Data import and column standardization**

The CSV file was loaded using latin1 encoding and semicolon separation. Original French column names were translated into English to improve readability and usability.

**3.2 Missing value analysis**

A missing-value audit showed that some pollutant variables had substantial missingness, especially:

* HC (g/km)
* HC+NOX (g/km)
* Particles (g/km)
* CO type 1 (g/km)
* NOX (g/km)

Consumption,  CO2 values had only a small number of missing observations.

<img width="1729" height="580" alt="Screenshot 2026-03-09 at 6 16 07 PM" src="https://github.com/user-attachments/assets/98df60fb-6d5e-4ccf-8836-6a10b50bac2c" />

**3.3 Pollutant reconstruction**

Missing pollutant values were partially reconstructed using relationships between:

HC+NOX = HC + NOX

This allowed the calculation of missing HC and NOX values where possible, improving dataset completeness.

**3.4 Correction of inconsistent categorical entries**

Some erroneous gearbox and regulatory field (Field V9) values were corrected, such as:
Gearbox:
* N 0, N 1 → A 0
* S 6 → D 6


**3.5 Electric vehicle adjustments**

For electric vehicles (Fuel == "EL"), several emissions and fuel-consumption values were replaced with 0, since these vehicles do not produce tailpipe CO2 or combustion pollutants in the same way as fuel-powered vehicles.

**3.6 Mass aggregation**

The variables Empty Mass Euro Min (kg) and Empty Mass Euro Max (kg) were merged into:

Empty Mass Euro Avg (kg)

This average mass was then used throughout the analysis.



# 4. Exploratory Data Analysis (EDA)

The exploratory analysis examined the structure and patterns of the dataset from several perspectives.  

## 4.1 Summary Statistics

A descriptive summary shows that the dataset contains **44,850 rows, 51 brands, 458 folder models**, and **13 fuel categories**. 

**The average values reported include:**  

* **Administrative Power:** about 11.02  
* **Maximum Power (kW):** about 124.78 kW  
* **Combined Consumption (l/100km):**  about 7.71 l/100km  
* **CO2 (g/km):**  about 198.74 g/km  
* **Avg Empty Mass (kg):** about 2120.25 kg  


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

The dataset is dominated by diesel (GO) vehicles, followed by gasoline (ES). Other categories such as electric and hybrid vehicles are present but much less frequent, reflecting the structure of the market at the time of data collection.


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

The dataset includes 51 brands, but the distribution is highly uneven. Mercedes-Benz has by far the most observations (~38,000), mainly due to very frequent models such as Viano, Vito, and Sprinter.

In terms of model diversity, Audi has the largest number of distinct models, followed by Volkswagen and Porsche, while Mercedes-Benz also offers a relatively wide range of models.

<img width="1638" height="784" alt="Screenshot 2026-03-10 at 2 24 47 PM" src="https://github.com/user-attachments/assets/1a86cf2c-b703-49db-a182-bc1b2246ba47" />

<img width="1619" height="776" alt="Screenshot 2026-03-10 at 5 37 44 PM" src="https://github.com/user-attachments/assets/ff978cd2-113e-4cf3-8f56-a7e5735a0b37" />

## 4.4 Top 50 Models by Number of Observations

<img width="1619" height="776" alt="Screenshot 2026-03-10 at 5 37 44 PM" src="https://github.com/user-attachments/assets/ff978cd2-113e-4cf3-8f56-a7e5735a0b37" />



## 4.4 CO2 Distribution

**Grouping data by Brand and shows brand-level averages for:**

* AvgCO2 (g/km)
* Avg Empty Mass (kg)
* Avg Maximum Power (kW)
* Avg Combined Consumption (l/100km)
* Vehicle Counts

**These aggregates are visualized in scatterplots showing**

* Average CO2 vs average maximum power by brand and number of observations
* Average CO2 vs average empty mass by brand and number of observations**


<img width="1638" height="784" alt="Screenshot 2026-03-10 at 2 25 33 PM" src="https://github.com/user-attachments/assets/78749ee5-6517-4ee0-9dac-f1e1037988c7" />


The dataset also includes a wide range of vehicles, from low to zero-emission brands to very high-emission luxury and performance brands.

**CO2 distribution by brand, fuel, and segment**

Boxplots showed substantial variation in CO2 emissions across Brand, Energy Types and Vehicle ranges. This indicates that emissions are strongly influenced by both engineering decisions and vehicle class.

<img width="1609" height="779" alt="Screenshot 2026-03-10 at 5 55 31 PM" src="https://github.com/user-attachments/assets/e0a14cc8-8ab4-4f0f-b464-71542025a7c1" />
<img width="1609" height="779" alt="Screenshot 2026-03-10 at 5 55 58 PM" src="https://github.com/user-attachments/assets/ca344226-24d6-4325-ba1f-5ce1c215b125" />
<img width="1609" height="779" alt="Screenshot 2026-03-10 at 5 56 55 PM" src="https://github.com/user-attachments/assets/ec3b9f63-4228-4a99-be42-0d1ff8d4ca69" />



# Exploratory Data Analysis (EDA)

-Examination of how different energy types are represented across vehicles.  
-Analysis of the distribution of vehicle manufacturers in the dataset.  
-Investigation of engine power–related variables.  
-Evaluation of CO₂ emission levels and their patterns.  
-Exploration of the relationships between vehicle weight, engine power, fuel consumption, and emissions.  
-Application of clustering methods to detect potential groupings based on vehicle weight and fuel consumption.   

# Key Findings
Some key findings from the analysis include:  

The distribution of energy types showed a significant presence of diesel and gasoline vehicles.  
Certain vehicle brands were more prevalent in the dataset.  
Power-related variables exhibited various distributions and relationships.  
CO2 emissions varied across different vehicle brands and energy types.  
One noteworthy observation is the strong correlation between weight and both consumption and CO2 emissions. Heavier vehicles tend to have higher consumption and emissions, emphasizing the importance of vehicle weight in environmental impact assessments.
