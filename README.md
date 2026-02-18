# Coding Challenge

This Code assessment represents a coding challenge for Data Engineering roles.

## Purpose

- Evaluate your coding abilities and your software engineering skills
- Judge your technical experience
- Understand how you design the solution 
- Have a technical coversation once the code submission is done

## Challenge 

In this challenge, we will use a dataset having the current job posting hosted by the City of New york which is available on the City of New York's official jobs site (https://www1.nyc.gov/jobs/index.page).

- Internal postings availble to city employees and external postings availbale to the public are included. 
- Data is accesible in the CSV file [nyc-jobs.csv](https://github.com/ProjectsForDataChapter/data_engineering_takehome1/blob/main/dataset/nyc-jobs.csv)

This challenge is composed of following steps: 

### Data Exploration

- Provide a detailed analysis of source data: Column values (eg: Numerical vs character), categorical columns, etc. 
- List of KPIs to be resolved:
  - Whats the number of jobs posting per category (Top 10)? 
  - Whats the salary distribution per job category? 
  - Is there any correlation between the higher degree and the salary?
  - Whats the job posting having the highest salary per agency? 
  - Whats the job positings average salary per agency for the last 2 years? 
  - What are the highest paid skills in the US market? 

### Data Processing
 
- Create functions to process your dataset (Cleaning, column pre-processing, data wrangling, transformation etc) 
- Apply atleast 3 feature engineering techniques 
- Features removal based on the exploration/ profiling.
- Store your processed data into a target file

## Expectations

- Above mentioned challenge
- Test cases 
- Code Comments 
- If any deployment to be done, proposals of the deployment steps
- If you had to trigger your code, please suggest your approach. 

## Coding Instructions

- PySpark should be used
- Feel free to use any libraries (you can use pip install, if needed)
- Use a visualization library to present your analysis results.
- Any learnings/ challenges/ considerations/ assumptions, please document in [MyDocument.md](https://github.com/ProjectsForDataChapter/data_engineering_takehome1/blob/main/MyDocument.md)

## Technical Support:

1. Clone the parent repository to your local. 
     - If git clone using password does not work follow the below steps:
       a. Click on your profile -> settings -> Developer settings -> Personal Access Tokens -> Tokens(Classic) -> Generate New Token
       b. In select scopes, give all repo permissions -> Click on Generate Token
       c. Copy the personal access token generated and copy it somewhere as it will be not visible again.
       d. In git terminal when you run git clone command (using HTTPS protocol) -> enter git id and for password enter the personal access token generated in previous step.
        
     - **Access to the parent repo will be removed within 4 days from the day you are made as collaborator**.
     
2. Push the cloned repo into **your own** Github account.
     - **Please don't commit anything to the parent repo, everything should be done in your own Github repo**.
     
3. Follow the installation instructions in [INSTALL.md](https://github.com/ProjectsForDataChapter/data_engineering_takehome1/blob/main/INSTALL.md)
4. Open the Jupyter notebook home page, using the link similar to the one highlighted in the picture 

  ![jupyter-notebook](https://github.com/ProjectsForDataChapter/data_engineering_takehome1/blob/main/pictures/docker-compose.png)

5. Open the notebook (assesment_notebook.ipynb). 
6. Read the documentation and run the cells accordingly. Once you create a spark session you should be able to see application **pyspark-assesment** running in the Spark Master UI as below. 

  ![spark-application](https://github.com/ProjectsForDataChapter/data_engineering_takehome1/blob/main/pictures/spark-application.png)

By now you would have already have below things in place:

- Created Spark cluster with a master and 2 worker nodes.
- Notebook to start working having the basic libraries imported (Please feel free to import any new library, if required). 
- Datasets read that is required for your activity.
- Sample function within the notebook arriving at a basic KPI.
- Sample test case for testing the function. 

## Submission:
 
Once you have completed your work, share an invite of **your own** GitHub repo to **projectsfordatachapter@gmail.com**, and send out an email to our hiring team. 
We will review your work.

# Happy Coding!!

