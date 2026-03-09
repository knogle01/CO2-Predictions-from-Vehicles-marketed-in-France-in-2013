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


**3.5. Electric vehicle adjustments**

For electric vehicles (Fuel == "EL"), several emissions and fuel-consumption values were replaced with 0, since these vehicles do not produce tailpipe CO2 or combustion pollutants in the same way as fuel-powered vehicles.

**3.6. Mass aggregation**

The variables Empty Mass Euro Min (kg) and Empty Mass Euro Max (kg) were merged into:

Empty Mass Euro Avg (kg)

This average mass was then used throughout the analysis.



# 4. Exploratory Data Analysis (EDA)

The exploratory analysis examined the structure and patterns of the dataset from several perspectives.

**4.1 Distribution of fuel types**

The dataset is dominated by diesel (GO) vehicles, followed by gasoline (ES). Other categories such as electric and hybrid vehicles are present but much less frequent, reflecting the structure of the market at the time of data collection.

<img width="1052" height="728" alt="Screenshot 2026-03-09 at 6 24 50 PM" src="https://github.com/user-attachments/assets/ddfbf583-03e9-41f3-adbd-a29e9497ed30" />

**Distribution of brands and models**

The dataset covers 51 brands and hundreds of models. Some brands (e.g MERCEDES-BENZ) appear much more frequently than others, and certain manufacturers also offer a wider variety of models.
**Distribution of Brands**
<img width="1210" height="702" alt="Screenshot 2026-03-09 at 6 27 02 PM" src="https://github.com/user-attachments/assets/751ae26c-b0cb-4566-a5bf-e9de642942a9" />
**Distribution of Models by Brand**
<img width="1212" height="791" alt="Screenshot 2026-03-09 at 6 29 14 PM" src="https://github.com/user-attachments/assets/1fcb5c3d-3026-4c6c-a3a5-69cdca963ce0" />

**Summary statistics**

Key numeric variables showed the following broad patterns:

Mean CO2 emissions: about 198.7 g/km
Mean maximum power: about 124.8 kW
Mean combined consumption: about 7.71 l/100km
Mean empty mass: about 2120 kg

The dataset also includes a wide range of vehicles, from low-emission compact cars to very high-emission luxury and performance vehicles.

**CO2 distribution by brand, fuel, and segment**

Boxplots showed substantial variation in CO2 emissions across:

Brands

Fuel types

Vehicle ranges / segments

This indicates that emissions are strongly influenced by both engineering decisions and vehicle class.




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
