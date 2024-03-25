<H1 align="center">Accident Severity Prediction to reduce Road Traffic Accidents in Seattle, WA</H1> 
                          <H2 align="center">IBM Capstone Project</H2>

![Seattle](https://github.com/AGK89/Coursera_Capstone/assets/153049066/6ed69a9d-854a-49ff-911c-fa9f4005e18e)

## Overview

This repository is home to the Coursera Capstone Project, an endeavor aimed at developing a supervised machine learning model with the capability to predict the severity of traffic accidents based on various circumstances. The project is part of a broader educational initiative, designed to apply and showcase the skills acquired through coursework in a practical, real-world problem-solving scenario.

## Project Objective

The primary goal of this project is to construct a predictive model that, when given data about specific conditions or factors surrounding an incident, can accurately forecast the accident's severity. This model seeks to leverage various datasets, employing statistical and machine learning techniques to understand the complex dynamics that contribute to the severity of accidents. Through this project, we aim to contribute valuable insights that could potentially aid in enhancing road safety measures and mitigating the impact of traffic accidents.

## Dataset

The dataset used in this project comprises detailed records of traffic accidents, encompassing a wide range of variables such as weather conditions, road conditions, time of day, and many others that are believed to have a significant impact on the outcome of accidents. The dataset is sourced from <a href="https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Metadata.pdf">here</a>, ensuring a robust foundation for model training and validation.

## Methodology

Our approach involves the following key steps:

1. **Data Preprocessing**: Cleaning and preparing the data for analysis, including handling missing values, outliers, and encoding categorical variables.
2. **Exploratory Data Analysis (EDA)**: Conducting a thorough analysis to understand the data's characteristics and uncover any underlying patterns or correlations between variables.
3. **Feature Engineering**: Creating new variables from the existing ones to improve the model's predictive power.
4. **Model Selection**: Evaluating various machine learning algorithms to identify the most suitable model based on performance metrics.
5. **Model Training and Validation**: Training the selected model on a subset of the data and validating its performance using a separate test set.
6. **Evaluation**: Assessing the model's accuracy, precision, recall, and F1 score to gauge its effectiveness in predicting accident severity.

## Technologies Used

- **Data Analysis and Modeling**: Python, Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Environment and Tools**: Jupyter Notebook, Git

## Installation

To set up your environment to run this project, follow these steps:

1. Clone this repository to your local machine using `git clone <repository-url>`.
2. Ensure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/).
3. Install the required Python libraries by running `pip install -r requirements.txt` in your command line.

## Usage

Instructions on how to run the project:

1. Navigate to the repository's root directory in your terminal.
2. Launch Jupyter Notebook by running `jupyter notebook`.
3. Open the project notebook (`capstone_project.ipynb`) in Jupyter and run the cells sequentially.


<!-- ## Table Of Contents:

* [Introduction](#Introduction)
* [Data](#Data)
* [Methodology](#Methodology)
* [Results](#Results)
* [Discussion](#Discussion)
* [Conclusion](#Conclusion)

# Introduction

Every year the lives of approximately 1.35 million people are cut short as a result of a road traffic crash. Between 20 and 50 million more people suffer non-fatal injuries, with many incurring a disability as a result of their injury.Road traffic injuries cause considerable economic losses to individuals, their families, and to nations as a whole. These losses arise from the cost of treatment as well as lost productivity for those killed or disabled by their injuries, and for family members who need to take time off work or school to care for the injured. Road traffic crashes cost most countries 3% of their gross domestic product.The study of influencing factors of traffic accidents is an important research direction in the field of traffic safety. The increasing number of crashes is a major public safety concern with various related costs. 

### Business Problem



In an effort to reduce the frequency of such collisions in the community, a model must be developed to predict the severity of an accident given the current weather, the road and visibility conditions. With our application, the user will be alerted to be more careful if the conditions are bad.

Our main objective in this project is to make a supervised prediction model that predicts the severity of an accident given certain circumstances (the current weather, road and visibility conditions) and alert the end user appropriately.<br>
<br>

# Data

This project will utilize Jupyter Notebooks to analyze a metadata set containing a rating of accident severity, street location, collision address type, weather condition, road condition, vehicle count, injuries, fatalities, and whether the driver at fault was under the influence. The dataset we will use in this project is the shared data originally provided by Seattle Department of Transportation(SDOT) Traffic Management Division, Traffic Records Group, and modified to particularly meet the project criteria.The dataset that we will be using is a .csv file named, 'Data-Collisions'. Our target variable will be 'SEVERITYCODE' because it is used to measure the severity of an accident from 0 to 3 (including a "2b", as per the metadata) within the dataset. Attributes that are used here to weigh the severity of an accident are 'WEATHER', 'ROADCOND' and 'LIGHTCOND'. The entire dataset originally had 194,673 rows (Instances) and 38 columns (Features). The metadata of the dataset can be found <a href="https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Metadata.pdf">here</a>. <br><br>In it's original form, this data is not fit for analysis. There are many columns that we will not use for this model. So, the data is to be cleaned, preprocessed and well prepared for analysis, and then to be fed to the Machine Learning Algorithms to finalize our model.

<br>

# Methodology

In this project, our main aim is to explore the relation between the road, light and weather conditions with respect to the accident severity in Seattle, WA. For this, the pre-processed data will be analyzed using Exploratory Data Analysis and Inferential Statistical Analysis. Based on the inference, we shall proceed with the selection of Machine Learning Algorithm for our model.  

### Exploratory Data Analysis <br>

The correlation Heat-Map of the dataset was explored. However, it did not provide much of an insight to the problem as our independent variables were shown to be Negatively correlated with the dependent variable. After that, the Pearson Coefficient and p-value were explored, which showed that the Road Condition and Light Condition had a strong relation with the Collision Severity. The initial decision of including the Weather Condition along with the Road and Light Condition was not changed.


<br>


### Machine Learning Algorithms & Evaluation <br>

**1. K-Nearest Neighbor (KNN)** <br>

The k-nearest neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems. It's easy to implement and understand, but has a major drawback of becoming significantly slows as the size of that data in use grows. Here we will be trying different values for k and get the result of the besk k-value which will be used to predict the output,i.e., KNN will help us predict the severity code of an outcome by finding the most similar data-point within k distance.<br>

![00](https://github.com/AGK89/Coursera_Capstone/assets/153049066/89a2da8b-4a3c-4706-b617-9a1eddfbbd49)
![01](https://github.com/AGK89/Coursera_Capstone/assets/153049066/a8244d6f-dd4f-4816-947b-42afd87c3871)
<br><br>

**2. Decision Tree** <br>

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. <br>
A decision tree model gives us a layout of all possible outcomes so we can fully analyze the concequences of a decision. In this context, the decision tree observes all possible outcomes of different weather conditions.<br>

![02](https://github.com/AGK89/Coursera_Capstone/assets/153049066/b19343a5-1202-4cbd-97b7-0ac993e69f47)

<br><br>

**3. Logistic Regression** <br>

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Unlike linear regression which outputs continuous number values, logistic regression transforms its output using the logistic sigmoid function to return a probability value which can then be mapped to two or more discrete classes. <br>
Because our current dataset (in this case) only provides us with two severity code outcomes, our model will only predict one of those two classes. This makes our classification binary, and logistic regression is a go-to method for binary classification problems, which makes it perfect for us.<br>

![03](https://github.com/AGK89/Coursera_Capstone/assets/153049066/60a8ded4-e00e-4a80-87a3-ddc068ed920e)

<br>
<br>

## Results

The dataset was fed to the Machine Learning Algorithms as mentioned in the Methodology Section. The accuracy of the 3 models are as shown below :<br>

![04](https://github.com/AGK89/Coursera_Capstone/assets/153049066/ef7bee60-c45a-43bb-a92a-6cd86eeacf16)

<br>
The above table shows that the Decision Tree Algorithm gives the highest accuracy of 56% (which 2% higher than Logistic Regression). 

# Discussion

In the beginning of this notebook, we had categorical data that was of type 'object'. This is not a data-type that we could have fed through an algoritim, so label encoding was used to created new classes that were of the type int (numerical data type).

After solving that issue we were presented with another - imbalanced data. As mentioned earlier, class 1 was nearly three times larger than class 2. The solution to this was downsampling the majority class. We downsampled to match the minority class exactly with 57052 values each.

Once we analyzed and cleaned the data, it was then fed through three ML models: K-Nearest Neighbor, Decision Tree and Logistic Regression. Although the first two are ideal for this project, logistic regression made most sense because of its binary nature.

Evaluation metrics used to test the accuracy of our models were Jaccard index, f-1 score and log_loss for logistic regression. Choosing different k, max depth and hyparameter C values helped to improve our accuracy to be the best possible.

**Future Scope**

We have just scratched the surface of this dataset with our use case. There is scope for a vast variety of analytics and modeling that can be done with this dataset for various other use cases (for example, finding out the relation between various alleys/intersections with collision severity in order to improve the infrastructure). 
<br>
By optimizing the dataset (multi-class outcomes instead of binary as in this case) and trying other algorithms, there is high scope for improvement of our model in the future.

# Conclusion

Based on the dataset featuring the weather, the road and lighting conditions, our model could predict the accident severity with an accuracy of 54%. We conclude that these accidents can be avoided if the end user is provided with real-time information on the road and lighting conditions and also regular updates on the weather using our application. -->
