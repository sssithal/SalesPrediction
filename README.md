Advertising Sales Prediction Using Linear Regression

This project explores the relationship between advertising expenditure on different media (TV, Radio, Newspaper) and resulting sales using a dataset advertising.csv. 
The project implements a linear regression model to predict sales based on TV advertising spend, using both statsmodels and scikit-learn.

Requirements:

    Python 3.x
    Libraries: numpy, pandas, matplotlib, seaborn, statsmodels, scikit-learn

Install required libraries with:

pip install numpy pandas matplotlib seaborn statsmodels scikit-learn

Dataset:

The advertising.csv file should contain the following columns:

    TV: Budget spent on TV advertisements
    Radio: Budget spent on radio advertisements
    Newspaper: Budget spent on newspaper advertisements
    Sales: Generated sales

Analysis and Modeling:

    Data Exploration
        Display the first few rows of the dataset.
        Use scatter plots and a correlation heatmap to visualize relationships between variables.
        Observations:
            TV advertising has a strong positive correlation with Sales.
            Radio advertising shows a moderate correlation.
            Newspaper advertising has a weak correlation with Sales.

    Linear Regression Model (statsmodels)
        Use Ordinary Least Squares (OLS) regression to find the line of best fit between TV advertising and Sales.
        Display key metrics from the regression summary, including coefficients, R-squared, and p-values.
        Observations:
            R-squared value of ~0.816 indicates that TV advertising accounts for around 81.6% of the variance in Sales.
            The low p-value for the TV coefficient suggests a statistically significant relationship between TV spend and Sales.

    Model Evaluation (Testing on New Data)
        Split the data into training and testing sets (70% training, 30% testing).
        Calculate Mean Squared Error (MSE) and R-squared for the predictions on test data to evaluate the model’s performance.

    Linear Regression Model (scikit-learn)
        Perform linear regression using LinearRegression from scikit-learn to cross-verify results.
        Extract the intercept and slope (coefficients) for the best-fit line.

Code Structure:

    Data Loading: Reads data from advertising.csv.
    Exploratory Data Analysis (EDA): Visualizes relationships and correlations.
    Model Building: Builds the regression model using statsmodels.
    Model Testing: Tests model accuracy using metrics.
    Alternate Model (scikit-learn): Builds a similar regression model using scikit-learn.

Running the Project:

    Place advertising.csv in the same directory as the script.
    Run the script:

    python advertising_sales_prediction.py

    The script will display data analysis visuals and output the model’s coefficients, intercept, R-squared values, and Mean Squared Error.

Results and Interpretation:

The linear regression model shows that TV advertising significantly predicts Sales, with a positive linear relationship. 
Both statsmodels and scikit-learn approaches yield consistent results.
