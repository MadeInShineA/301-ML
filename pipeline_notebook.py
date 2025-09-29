import marimo

__generated_with = "0.16.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Data Pipeline - (Machine) Learning from Disaster

    In this notebook, you will implement a full machine learning workflow, from preprocessing to model evaluation. 
    The dataset we use is a famous one: [the Titanic dataset](https://www.kaggle.com/competitions/titanic/data) (yes, the big boat that sank...).

    The idea is simple: use ML to predict which passengers **survived** the Titanic shipwreck. The dataset is quite simple to understand but presents some real-world challenges (e.g., missing values). The explanation of the dataset is available at the link above.

    ## Goals
    The *first* and most important goal of this lab is to guide you towards a higher level of autonomy when dealing with ML problems, in particular, classification problems (and in the later part, optionally, you will deal with a regression problem). 

    The *second* goal is to get an understanding of the Support Vector Machine (SVM) algorithm, which is a (relatively) simple but powerful ML algorithm.

    This document provides just the skeleton of your program, reminding you of the main steps to be accomplished.
    At the end of this lab, you will be able to:

    - Work on a jupyter notebook for a ML problem.
    - Develop a full Machine Learning pipeline starting from a skeleton.
    - Perform data exploration and data preparation
    - Train, tune and **properly** evaluate ML models.

    ## Structure
    This notebook is divided into 3 main parts:

    1. **Data Exploration**: where you will explore the dataset using the Pandas library.
    2. **Data Preparation**: where you will preprocess the data to be used in the ML model.
    3. **Modeling, fine-tuning and evaluation**: where you will train, tune and evaluate the ML models.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1 Data Exploration""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1.1 Read the data
    We dowloaded the dataset for you. You will find the .csv file in the `data` folder.
    """
    )
    return


@app.cell
def _():
    # some useful imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo

    # read csv from data folder
    df_train = pd.read_csv('data/train.csv')
    return df_train, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the cell above, we read and stored the data in a pandas *dataframe*.
    A dataframe is a 2-dimensional labeled data structure with columns of potentially different types.
    You can think of it like a spreadsheet or SQL table, or a dictionary of Series objects.

    In the next cells, we will start exploring the data while starting to learn how to use the most important functions of the pandas library.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1.2 Have a look to the data

    In Machine Learning, the **first** thing we want to do is to look at the data. *Always*.
    This page https://www.kaggle.com/competitions/titanic/data contains the so-called "metadata".
    Metadata refers to data that provides information about other data. It describes various aspects of data, such as its origin, format, structure, and context, making it easier to understand, manage, and use. Metadata can be crucial for organizing, locating, and interpreting different types of information within a dataset or system.

    In the metadata description you can find information about the data we are going to use. Give it a look.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Question**:

    Checking the data card of the model, how many features are there? How many labels? (or target?). In other terms, if you would like to put this data in a tabel-like structure, how many rows would you have? How many columns?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answer**: 

    Assuming passengerID and name are not features, there are 9 features and 1 target. So, 11 columns in total. There is 1 row per passenger. In total there is len(df_train) + len(df_test) rows.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's see if the data are consistent with the metadata.
    To do that, we will use our first pandas function: `head()`.
    """
    )
    return


@app.cell
def _(df_train):
    # we print the first 5 rows of the dataframe
    df_train.head()
    return


@app.cell
def _(df_train):
    # we print the first 10 rows of the dataframe
    df_train.head(10)
    return


@app.cell
def _(df_train):
    # we also print the last 5 rows of the dataframe
    df_train.tail()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To get a better understanding of the data, we can also print the shape of the dataframe (i.e., number of rows, columns, etc.), the column names and data types.
    Finally, we can also print the descriptive statistics of the data.
    This is done using the functions `shape`, `info` and `describe`, respectively.
    """
    )
    return


@app.cell
def _(df_train):
    # we print the shape of the dataframe
    df_train.shape
    return


@app.cell
def _(df_train):
    # we print the column names and data types
    df_train.info()
    return


@app.cell
def _(df_train):
    # we print the descriptive statistics of the dataframe
    df_train.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:

    1. Is the data consistent with the metadata?
    2. Which columns are features? Which columns are labels? (or target?)
    3. Among the features, there is one that is not correlated with any possible useful prediction. Which one and why?
    4. What is the mean of the target?
    5. Why not all columns are present in the descriptive statistics?
    6. Is the dataset in your opinion "balanced" with respect to the column `Survived` (hint: try the method `df['Survived'].value_counts()`) ? If not, can this be a problem?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answer**:

    1. The metadata doesn't mention the PassengerId and Name columns, but otherwise the data seems consistent with the metadata.
    2. Features : Pclass, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
       Target : Survived
    3. # TODO
    4. 0.383838
    5. Only numerical columns are present in the descriptive statistics.
    6. The dataset is not balanced with respect to the column Survived. There are 549 non-survivors and 342 survivors. This can be a problem because the model may be biased towards the majority class (non-survivors) and may not perform well on the minority class (survivors).
    """
    )
    return


@app.cell
def _(df_train):
    # we print the value counts of the column Survived to check for imbalance
    df_train['Survived'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1.3 Missing data
    In this subsection we check if our dataset is complete.
    """
    )
    return


@app.cell
def _(df_train):
    # we print the number of missing values in the dataframe
    df_train.isnull().sum()
    return


@app.cell
def _(df_train):
    # we check the percentage of missing values in the dataframe
    df_train.isnull().mean() * 100
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:

    - Are there missing values in the dataset?
    - If yes, how many missing values are there in each column?
    - Are there missing values in the target column? If yes, how many?

    NOTES: Knowing the purcentage of missing values and their type (i.e., numerical, categorical, etc.) is crucial for deciding how to handle them.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**:
    - Are there missing values in the dataset? Yes
    - If yes, how many missing values are there in each column?
        - PassengerId: 0
        - Survived: 0
        - Pclass: 0
        - Name: 0
        - Age: 177
        - SibSp: 0
        - Parch: 0
        - Ticket: 0
        - Fare: 0
        - Cabin: 687
        - Embarked: 2
    - Are there missing values in the target column? If yes, how many? No
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1.4 Correlation analysis
    In this subsection, we check the correlation among features and between features and the target. We will plot a correlation matrix and a heatmap.

    NOTE: For simplicity, we will only consider numerical features. We could also consider categorical features by converting them to numerical ones (e.g., using one-hot encoding, see next section).
    """
    )
    return


@app.cell
def _():
    # list of numerical variables
    numerical_features = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

    # list of categorical variables
    categorical_features = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    return (numerical_features,)


@app.cell
def _(df_train, numerical_features):
    # we select the numerical columns 

    df_numerical = df_train[numerical_features]
    return (df_numerical,)


@app.cell
def _(df_numerical, plt):
    # we plot the correlation matrix of the numerical variables (it can take some time, depending on the size of the data)
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    correlation_matrix = df_numerical.corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.show()
    return (sns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:

    1. What is the most correlated feature with the target `Survived`? What does it means?
    2. What is the less correlated feature with the target `Survived`? Is that a surprise?
    3. What is the most correlated feature with the feature `Pclass`?
    4. Is it good to have highly correlated features in the dataset? Why?
    5. What is the meaning of a negative correlation?
    6. What is the meaning of high correlation between a feature and the target?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**:

    1. Pclass
    2. PassengerId < SibSp
    3. Fare
    4. It's not good because it gives to much importance to an information that is already present in another feature.
    5. A negative correlation means that as one variable increases, the other variable tends to decrease. In other words, there is an inverse relationship between the two variables.
    6. High correlation between a feature and the target indicates that the feature is a strong predictor of the target variable. This means that changes in the feature are closely associated with changes in the target, making it a valuable variable for building predictive models.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Extra - Getting used to pandas

    This is not part of the usal ML pipeline, but it is important to get used to pandas. In the previous cells, we learned how to read data from a csv file and how to get a first look at the data.

    In this subsection, we will practice some pandas commands that will help us in the next sections and labs.
    In particular, we will learn how to select columns and rows from a dataframe and how to filter data.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### E.1 Selecting columns using columns names""")
    return


@app.cell
def _(df_train):
    # we select the column 'Ticket' and store it in a variable called ticket;
    # ticket is a pandas series (you can think of it as a 1-dimensional array)
    ticket = df_train['Ticket']

    # we print the first 5 rows of the air_temperature variable
    ticket.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the previous example, we selected a column from the dataframe.
    We can now try to select multiple columns from the dataframe.
    """
    )
    return


@app.cell
def _(df_train):
    # we select the columns 'Ticket' and 'Fare' and store them in a variable called ticket_and_fare;
    # ticket_and_fare is a pandas dataframe (you can think of it as a 2-dimensional array)
    ticket_and_fare = df_train[['Ticket', 'Fare']]

    # we print the first 5 rows of the temperatures variable
    ticket_and_fare.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### E.2 Selecting rows and columns using their index (iloc)
    We can also select rows from the dataframe.
    `iloc` is a method that allows us to select rows from the dataframe. It returns a row as a pandas series or a dataframe (if more than one raw are selected).
    """
    )
    return


@app.cell
def _(df_train):
    # we select the first row of the dataframe
    df_train.iloc[:0]
    return


@app.cell
def _(df_train):
    # we select multiple rows from the dataframe
    df_train.iloc[[0, 1, 2]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Like lists, numpy arrays, etc. it is possible to "slice" pandas series and dataframes. We use the method `.iloc()` Here are some examples:""")
    return


@app.cell
def _(df_train):
    # we select the first 5 rows of the dataframe
    df_train.iloc[:5]
    return


@app.cell
def _(df_train):
    # we select the last 5 rows of the dataframe
    df_train.iloc[-5:]
    return


@app.cell
def _(df_train):
    # we select the rows from index 5 to index 10
    df_train.iloc[5:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""With `iloc`, we can also select columns from the dataframe. Here are some examples:""")
    return


@app.cell
def _(df_train):
    # we select the columns from index 0 to index 2
    df_train.iloc[:, 0:2]
    return


@app.cell
def _(df_train):
    # we select the rows from index 5 to index 10 and the columns from index 0 to index 2
    df_train.iloc[5:10, 0:2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Question**:

    1. Select the first 5 rows of the dataframe and the columns from index 0 to index 2.
    2. Select the last 5 rows of the dataframe and the columns from index 0 to index 2.
    3. Select the rows from index 15 to index 25 and the columns from index 2 to index 3.

    *Suggestion: do not hesitate to add multiple cells to answer the questions.*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Answers**:""")
    return


@app.cell
def _(df_train):
    # 1. Select the first 5 rows of the dataframe and the columns from index 0 to index 2.
    df_train.iloc[:5, 0:2]
    return


@app.cell
def _(df_train):
    # 2. Select the last 5 rows of the dataframe and the columns from index 1 to index 3.
    df_train.iloc[-5:, 1:3]
    return


@app.cell
def _(df_train):
    # 3. Select the first 5 rows of the dataframe and the 'Ticket' and 'Fare' columns.
    df_train.iloc[:5][['Ticket', 'Fare']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### E.3 Selecting rows and columns using their labels (a.k.a. name) (loc)

    `iloc()` is not the only method that allows us to select rows from a dataframe.
    We can also use `loc()`. The difference between `iloc()` and `loc()` is that `iloc()` selects rows by index and `loc()` selects rows by label. Here are some examples:
    """
    )
    return


@app.cell
def _(df_train):
    # we select the first row of the dataframe
    df_train.loc[0]

    # we select multiple rows from the dataframe
    df_train.loc[[0, 1, 2]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As you can see there is not much of a difference when working with *rows*, since rows are *usually* indexed with numbers.
    However, when working with **columns**, `loc()` is arguably more useful (and readable) than `iloc()`.
    """
    )
    return


@app.cell
def _(df_train):
    # we select the column 'SibSp' using loc
    df_train.loc[:, 'SibSp']
    return


@app.cell
def _(df_train):
    # we select the columns 'SibSp' and 'Sex' using loc
    df_train.loc[:, ['SibSp', 'Sex']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:

    1. Select the first 5 rows of the dataframe and the column `Ticket` using loc.
    2. Select 10 rows starting at the index 11 for the columns`Fare` and `Sex` using loc.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Answer**:""")
    return


@app.cell
def _(df_train):
    # 1. Select the first 5 rows of the dataframe and the column `Ticket` using loc.
    df_train.loc[:5, 'Ticket']
    return


@app.cell
def _(df_train):
    # 2. Select 10 rows starting at the index 11 for the columns`Fare` and `Sex` using loc.
    df_train.loc[11:21, ['Fare', 'Sex']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**NOTE 1**:  test1 = df.loc[:, 'Fare'] is equivalent to test2 = df['Fare']""")
    return


@app.cell
def _(df_train):
    # 3. Is **test1** = df.loc[:, 'Fare'] equivalent to **test2** = df['Fare']? 
    test1 = df_train.loc[:, 'Fare']
    test2 = df_train['Fare']
    test1.equals(test2)
    return test1, test2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **NOTE 2**: by default we are **NOT** creating new dataframes. Dataframes maybe very big, copying them will take too much resources. In the previous exemple, `test1` and `test2` are pointing at the *same* address in the memory. In other terms, by default, we work with "views" on the same dataset.
    You can check this with the `is` operator:
    """
    )
    return


@app.cell
def _(test1, test2):
    if test1 is test2:
        print("yup, told ya, the two variables point to the same object in memory")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**For the sake of completeness**: you may need to copy a dataframe. If you want to duplicate a dataframe, you have to use the method `.copy()`, see the [official documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html) for more information.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### E.4 Filtering data
    Last but not least, with pandas we can also easily filter data. Filtering data means selecting rows that satisfy a certain condition.
    """
    )
    return


@app.cell
def _(df_train):
    # we select the rows where the column 'Sex' is equal to 'male' (i.e., we only select male passengers)
    df_train[df_train['Sex'] == 'male']
    return


@app.cell
def _(df_train):
    # we select the rows where the column 'Pclass' is equal to 2 (i.e, we select the passenger travelling in second class)
    df_train[df_train['Pclass'] == 2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:

    1. Select the passengers travelling in 3rd class.
    2. Select the passengers older than 18
    3. (hard) Select the *famale* passengers with strinctly more than 1 parent / children aboard the Titanic
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Answer**:""")
    return


@app.cell
def _(df_train):
    # 1. Select the passengers travelling in 3rd class.
    df_train[df_train['Pclass'] == 3]
    return


@app.cell
def _(df_train):
    # 2. Select the passengers older than 18 (18 included).
    df_train[df_train['Age'] >= 18]
    return


@app.cell
def _(df_train):
    # 3. Select the female passengers with strictly more than 1 parent / children aboard the Titanic
    df_train[(df_train['Sex'] == "female") & (df_train['Parch'] > 1)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1.3 Conclusion Section 1
    In this Section, we learned how to read data from a csv file and how to get a look at the data using the Pandas library. 
    We also learned how to select columns and rows from a dataframe using the Pandas library and how to filter data using the Pandas library.

    Today, Pandas is almost ubiquitous in machine learning when coding in Python. If the methods we’ve covered so far aren’t yet clear to you, don’t worry—it will become easier as you practice these techniques.

    In the next secgtion, we will learn how to prepare the data for the training using another famous library: **Scikit-learn**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""---""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Data preparation and introduction to Scikit-learn

    In this second notebook, our primary emphasis will be on data preparation. This involves the essential steps of dividing the dataset into a *training set* and a *test set*, along with gaining insights into best practices for addressing specific issues or characteristics within our data. These characteristics may include:

    - Features with varying scales
    - The intermixing of numerical features (e.g., `Age`) and categorical features (e.g., `Sex`)
    - Handling missing values

    To tackle these challenges, we will use [Scikit-learn](https://scikit-learn.org/stable/). Scikit-learn, also known as sklearn, is an open-source machine learning library for the Python programming language. It is built on NumPy, SciPy, and Matplotlib, and it provides simple and efficient tools for data analysis and modeling. Scikit-learn is designed to work with a variety of machine learning tasks, including classification, regression, clustering, dimensionality reduction, and more.

    Scikit-learn is widely used in both academia and industry for its user-friendly interface, extensive documentation, and the wealth of tools it offers for building and evaluating machine learning models.


    **Goals**

    In this second Section, we will focus on data preparation. In particular, you will learn how to use the Scikit-learn library to:
    - Split the data into training and test sets
    - Impute missing values
    - Scale and/or standardize the data
    - Encode categorical features

    Remember: all the insights used for data preparation (e.g., values and logic used for imputation, scaling, etc.) should be learned from the training set **only**. Then, the same approach should be applied to training and test set. This is crucial to avoid data leakage and to ensure that the model is trained and evaluated on the same data distribution.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.1 Separate the labels from the features""")
    return


@app.cell
def _(df_train):
    # we print (again) the first rows of the dataframe (for recap and better visualization)
    df_train.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Checking the columns, and with the information we got drom the Data Exploration section, we separate the columns into multiple lists:
    - useless_columns: columns that are not useful for the model (e.g., `PassengerId`)
    - numerical_columns: columns that are numerical (e.g., `Age`, `Fare`)
    - categorical_columns: columns that are categorical (e.g., `Sex`, `Embarked`)
    - target_column: the column we want to predict (e.g., `Survived`)
    """
    )
    return


@app.cell
def _():
    usless_columns = ['PassengerId', 'Name']
    target = 'Survived'
    numerical_features_1 = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    categorical_features_1 = ['Sex', 'Embarked']
    return categorical_features_1, numerical_features_1, target, usless_columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Before splitting the data into *training* and *test sets*, we need to separate the *labels* from the *features*. It is of **paramount importance** not providing the labels to the model during training!

    This is the *noobest* mistake you can do in ML. It is like providing a test with the solutions: the model will learn the right solution "by heart" without actually understanding the data. Then, when you will provide new data, with no solution... good luck with that!

    NOTES:

    - In ML, the letter `X` usually indicates the features and `y` the labels. Here, we will use `X` and `y` to store the features and labels, respectively. We can do this first separation by using the `drop()` method of the Pandas library.
    - the `drop()` method returns a **copy** of the dataframe with the specified columns removed. The `drop()` method does not change the original dataframe.
    - since we are dropping columns, we will also drop usless columns (e.g., `PassengerId`). We will also drop the column `Cabin` because it has too many missing values (which was a precious insight from the Data Exploration section).
    """
    )
    return


@app.cell
def _(df_train, target):
    # we select the target to create our target variable y
    y = df_train[target]

    # we drop the target from the dataframe to create our features X
    X = df_train.drop(columns=[target])
    return X, y


@app.cell
def _(X):
    # we print the first 5 rows of the features
    X.head()
    return


@app.cell
def _(y):
    # we print the first 5 rows of the labels
    y.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""From the insights that we got in the previous section, we drop the features that we think are not useful for our analysis.""")
    return


@app.cell
def _(X, usless_columns):
    X_1 = X.drop(columns=usless_columns)
    return (X_1,)


@app.cell
def _(X_1):
    X_1.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    NOTEs:

    - there is a missing value in the raw 5 for the column `Age`.
    - now that we splitted the data into features and labels, we need to be careful in the next steps. Some methods require shuffling the data: if you change the order of the features, you will have to change the order of the labels accordingly!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.2 Split the data into training and test sets""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""It is now time to actually split the data into "Training" and "Test" set. Remember: the test set will only be used for testing our model. NO learning should use these data. Once splitted, it is like these data do not exist for you until the final evaluation step.""")
    return


@app.cell
def _(X_1, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, df_train, y_test, y_train):
    # we check the shape of the original dataset
    print('The shape of the original dataset is:', df_train.shape)

    # we check the shape of the training set (features)
    print('The shape of the training set is:', X_train.shape)

    # we check the shape of the test set (features)
    print('The shape of the test set is:', X_test.shape)

    # we check the shape of the training set (labels)
    print('The shape of the training labels is:', y_train.shape)

    # we check the shape of the test set (labels)
    print('The shape of the test labels is:', y_test.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:

    - What is the purpose of the `test_size` parameter?
    - What is the purpose of the `random_state` parameter? Should I change this parameter to find the one that gives the best results?
    - Do we need to shuffle the data? Why is it important to shuffle (or not) the data before splitting it? (Hint: check in the doc the default value of the `shuffle` parameter)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**: 

    - The `test_size` parameter determines the proportion of the dataset to include in the test split. For example, if `test_size=0.2`, then 20% of the data will be used for testing and 80% for training.
    - The `random_state` parameter controls the shuffling applied to the data before applying the split. It ensures that the split is reproducible. If you set a specific value for `random_state`, you will get the same split every time you run the code. You should not change this parameter to find the one that gives the best results. Instead, you should use cross-validation to evaluate your model.
    - Yes, we need to shuffle the data. It is important to shuffle the data before splitting it to ensure that the training and test sets are representative of the overall dataset. If the data is not shuffled, the training and test sets may be biased, leading to poor model performance.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2.3 Impute missing values

    In this section, we will learn how to impute missing values using the Scikit-learn library.

    Imputation

    - Definition: the process of replacing missing data with substituted values.
    - Example: we have missing values about the "age" of some passengers. Instead of dropping the whole raw of data, we substitute the missing age value with the average value of the age in the trining set.
    - Be aware:
        - This is a trade-off: we introduce a bias in the data instead of losing some information.
        - Depending on some factors such as the size of the original dataset, the amount of missing data, this can be a good or a bad idea. If you have a big dataset, losing some data is not a big deal. If you have a small dataset, losing some data can be a big deal.
        - Never impute missing values in the target (y). You should drop the rows with missing values in the target.

    Missing values

    Missing values are often represented as `NaN` (Not a Number) or `None`. In the code below, we use the `isnull()` method of the Pandas library to check if there are any missing values in the dataset. The `isnull()` method returns a dataframe of booleans that indicate whether or not a value is missing (`True`) or not missing (`False`).
    Then, we use the `any()` method of the Pandas library to check if the dataframe contains any `True` values. If the dataframe contains any `True` values, then we know that there are missing values in the dataset.

    **Note (again)**: remember to perform all these tests on the training set ***only***. Always consider as the test set **is not** available during the training phase.
    """
    )
    return


@app.cell
def _(X_train):
    # Just for recap, we print again the missing values in the training set
    # # we check if there are any missing values in the training set
    missing_values_training_X = X_train.isnull()

    # we check if test contains any True values
    print(missing_values_training_X.any())

    # alternatively, if you want to see the number of missing values per feature
    # we check the percentage of missing values in the dataframe
    print()
    print(missing_values_training_X.mean() * 100 )
    return


@app.cell
def _(y_train):
    # Just for recap, we print again the missing values in the training set
    # # we check if there are any missing values in the training set
    missing_values_training_y = y_train.isnull()

    # we check if test contains any True values
    print(missing_values_training_y.any())

    # alternatively, if you want to see the number of missing values per feature
    # we check the percentage of missing values in the dataframe
    print()
    print(missing_values_training_y.mean() * 100 )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As we already observed, there are missing values in the columns `Age`, and `Embarked` (we removed `Cabin`).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Question**:
    - Repeat the test above to check if there are any missing values in the test set.
    """
    )
    return


@app.cell
def _(X_test):
    # we repeat the test for the test set. In this case, we can expect the same missing values in the test set as in the training set.
    # however, it is good practice to check it anyway (i.e., in some cases the test set may contain missing values that are not present in the training set)
    missing_values_test_X = X_test.isnull()

    print(missing_values_test_X.any())

    print()
    print(missing_values_test_X.mean() * 100 )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let us start with the imputation

    Some algorithms are able to handle missing values without preprocessing. You can find a list of these algorithms [here](https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values). However, most algorithms are not designed to handle missing values. Therefore, we need to impute them before training the model. There are many ways to impute missing values.

    Possible approaches:
    1. Drop rows with missing values. If you have a lot of data, this could you best option.
    2. Impute missing values with the "mean" or "median". This apporach is suitable for numerical features.
    3. Impute missing values with the "mode" (i.e., the most frequent value).  This apporach is suitable for categorical features (computing the mean does not make sense).
    4. Impute missing values using more advanced approaches (such as k-nn!)

    As an exemple, we will see how to implement 2 and 3 using the Scikit-learn library. 

    In our dataset, we have both numerical features (e.g., `Age`) and categorical features (e.g., `Embarked`) with missing values. We will need to deal with them separately.
    """
    )
    return


@app.cell
def _(X_test, X_train, numerical_features_1):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train[numerical_features_1])
    train_set_imputed_num = imputer.transform(X_train[numerical_features_1])
    test_set_imputed_num = imputer.transform(X_test[numerical_features_1])
    return SimpleImputer, test_set_imputed_num, train_set_imputed_num


@app.cell
def _(SimpleImputer, X_test, X_train, categorical_features_1):
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_cat.fit(X_train[categorical_features_1])
    train_set_imputed_cat = imputer_cat.transform(X_train[categorical_features_1])
    test_set_imputed_cat = imputer_cat.transform(X_test[categorical_features_1])
    return test_set_imputed_cat, train_set_imputed_cat


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We test again the data to see if there are still some missing values.

    NOTE: the Scikit-learn `.transform()` method returns a numpy.ndarray not a data frame. To continue working with pandas we need to convert it back to a data frame. However, for the moment, we will keep it as a numpy array since also the next steps require numpy arrays.
    """
    )
    return


@app.cell
def _(train_set_imputed_num):
    # we print the first 10 rows of the training set (there should no be more missing values in the sixth row)
    train_set_imputed_num[:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Optional task
    Scikit-lern offers more advanced imputers. For example, the `IterativeImputer` class models each feature with missing values as a function of other features, and uses that estimate for imputation.
    The `KNNImputer` class imputes missing values using the k-Nearest Neighbors approach (yes, we can use ML to solve ML problems).
    For more information on these and other classes, see the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/impute.html#impute).

    Try the `IterativeImputer` class and the `KNNImputer` class to impute the missing values in the training set. Then, check if there are any missing values in the training set.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2.4 Scale and/or standardize the data
    Rescaling or standardizing the data is useful for many algorithms that suffers when features' scale changes (e.g., KNN, SVM, Neural Networks, etc.). Basically, this concerns all algorithms based on the distance between data points.
    Other algorithms, such as Decision Trees, are not affected by the scale of the features.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the code below, we use the `StandardScaler` class to scale and/or standardize the data. The `StandardScaler` class standardizes features by removing the mean and scaling to unit variance. The standard score of a sample `x` is calculated as: 

    $$
    z = (x - \mu) / \sigma
    $$

    where $\mu$ is the mean of the training samples and $\sigma$ is the standard deviation of the training samples.
    """
    )
    return


@app.cell
def _(test_set_imputed_num, train_set_imputed_num):
    # from the sklearn library we import the function to scale and/or standardize the data
    from sklearn.preprocessing import StandardScaler # other options include MinMaxScaler, RobustScaler, etc.

    # we create a StandardScaler object
    scaler = StandardScaler()

    # we fit the scaler to the training set
    scaler.fit(train_set_imputed_num)

    # we transform the training set
    train_set_scaled_num = scaler.transform(train_set_imputed_num)

    # we transform the test set
    test_set_scaled_num = scaler.transform(test_set_imputed_num)
    return test_set_scaled_num, train_set_scaled_num


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:

    - We only rescaled the numerical features. Why did we not rescale the categorical features?
    - Compare the standardized values with the original values. Compute the mean (`np.average()`) and the standard deviation (`np.std()`) of the original and standardized values. What do you expect and what do you observe?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**:

    - We did not rescale the categorical features because they are not numerical. Rescaling categorical features does not make sense. For example, if we have a categorical feature with values `['A', 'B', 'C']`, rescaling these values would not make sense. The values are not ordered, and the distance between them is not meaningful. However, Categorical features should be encoded before being used in the model.
    - You should see that the standardized values have a mean of 0 and a standard deviation of 1.
    """
    )
    return


@app.cell
def _(train_set_scaled_num):
    # we print the average of the scaled training set
    print('The average of the scaled training set is:', train_set_scaled_num.mean(axis=0))

    # we print the standard deviation of the scaled training set
    print('The standard deviation of the scaled training set is:', train_set_scaled_num.std(axis=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Scikit-learn provides a number of other classes for scaling and/or standardizing the data. For example, the `MinMaxScaler` class scales features to a given range, usually between 0 and 1. The `RobustScaler` class scales features using statistics that are robust to outliers. For more information on these and other classes, see the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:

    - instead of using the `StandardScaler` class, use the `RobustScaler` class to scale and/or standardize the data. Then, print the first 5 rows of the scaled training set.
    - What is the difference between the `StandardScaler` and the `RobustScaler`? When should you use one over the other?
    """
    )
    return


@app.cell
def _(test_set_imputed_num, train_set_imputed_num):
    from sklearn.preprocessing import RobustScaler

    robustScaler = RobustScaler()

    # fit the scaler to the training set
    robustScaler.fit(train_set_imputed_num)

    # transform the training set
    train_set_scaled_num_2 = robustScaler.transform(train_set_imputed_num)

    # transform the test set
    test_set_scaled_num_2 = robustScaler.transform(test_set_imputed_num)

    # print the 5 first rows of the training set
    print(train_set_scaled_num_2[:5])
    return


@app.cell
def _(train_set_scaled_num):
    # print the 5 first rows of the training set
    print(train_set_scaled_num[:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**:

    - StandardScaler : mean=0, variance=1
    - RobustScaler : centered around the median and uses the interquartile range for scaling.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2.5 Encode categorical features
    In the previous section, we saw how to scale and/or standardize the numerical features. In this section, we will learn how to encode the categorical features.

    Some algorithms do not have any problem to deal with a mixed feature set (i.e., numerical and categorical features). However, other algorithms require that all the features are numerical. Therefore, we need to encode the categorical features before training the model.
    Typically, distance-based algorithms (e.g., KNN, SVM, Neural Networks, etc.) require that all the features are numerical.
    How could you compute the distance between a numerical feature and a categorical feature? It does not make sense, right? Therefore, we need to encode the categorical features before training the model.

    There are many ways to encode categorical features. We use the `OneHotEncoder` class to encode the categorical features. The `OneHotEncoder` class encodes categorical features as a one-hot numeric array. A one-hot array is a binary matrix where each row has exactly one element set to 1 and all other elements set to 0.
    """
    )
    return


@app.cell
def _(categorical_features_1, pd, test_set_imputed_cat, train_set_imputed_cat):
    X_train_set_imputed_cat = pd.DataFrame(train_set_imputed_cat, columns=categorical_features_1)
    X_test_set_imputed_cat = pd.DataFrame(test_set_imputed_cat, columns=categorical_features_1)
    return X_test_set_imputed_cat, X_train_set_imputed_cat


@app.cell
def _(X_test_set_imputed_cat, X_train_set_imputed_cat):
    # we import the OneHotEncoder class from the sklearn library
    from sklearn.preprocessing import OneHotEncoder

    # we create an OrdinalEncoder object
    encoder = OneHotEncoder(sparse_output=False) # we set sparse_output=False to get a 2D array

    # we fit the encoder to the training set
    encoder.fit(X_train_set_imputed_cat)

    # we transform the training set
    X_train_set_encoded = encoder.transform(X_train_set_imputed_cat)

    # we transform the test set
    X_test_set_encoded = encoder.transform(X_test_set_imputed_cat)
    return X_test_set_encoded, X_train_set_encoded, encoder


@app.cell
def _(
    X_test_set_encoded,
    X_train_set_encoded,
    categorical_features_1,
    encoder,
    pd,
):
    X_train_set_encoded_df = pd.DataFrame(X_train_set_encoded, columns=encoder.get_feature_names_out(categorical_features_1))
    X_test_set_encoded_df = pd.DataFrame(X_test_set_encoded, columns=encoder.get_feature_names_out(categorical_features_1))
    X_train_set_encoded_df.head()
    return X_test_set_encoded_df, X_train_set_encoded_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Question**
    - Other encoders exist such as the `OrdinalEncoder`. Explain the difference between the `OrdinalEncoder` and the `OneHotEncoder`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answer**:
    - OrdinalEncoder : one column per feature, values are integers representing the categories (0, 1, 2, ...)
    - OneHotEncoder : one column per category, values are binary (0 or 1)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2.6. Encode the target (label)

    Finally, in some cases, we need to encode the target (label) as well. Suppose you have to classify "apple", "banana", and "orange". You can encode them as 0, 1, and 2, respectively. This is useful when you have a classification problem with more than two classes. However, in our case, we have a binary classification problem (i.e., survived or not survived). Therefore, we do not need to encode the target. To get more information about encoding the target, see the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/preprocessing_targets.html#preprocessing-targets).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2.7. Putting it all together

    As a result of steps 5 and 6 we create sub-datasets of the original dataset. In particular, we have:
    - `X_train_set_scaled`: the training set with scaled and/or standardized numerical features
    - `X_test_set_scaled`: the test set with scaled and/or standardized numerical features
    - `X_train_set_encoded`: the training set with encoded categorical features
    - `X_test_set_encoded`: the test set with encoded categorical features
    - `y_train_set_encoded`: the training labels with encoded categorical features
    - `y_test_set_encoded`: the test labels with encoded categorical features

    Before starting the training and the evaluation in the next section, we need to concatenate the scaled and/or standardized numerical features with the encoded categorical features (we want a pandas dataframe).
    """
    )
    return


@app.cell
def _(numerical_features_1, pd, test_set_scaled_num, train_set_scaled_num):
    X_train_set_scaled = pd.DataFrame(train_set_scaled_num, columns=numerical_features_1)
    X_test_set_scaled = pd.DataFrame(test_set_scaled_num, columns=numerical_features_1)
    return X_test_set_scaled, X_train_set_scaled


@app.cell
def _(
    X_test_set_encoded_df,
    X_test_set_scaled,
    X_train_set_encoded_df,
    X_train_set_scaled,
    pd,
):
    # we concatenate the numerical and categorical dataframes to get the final training and test sets
    X_train_final = pd.concat([X_train_set_scaled, X_train_set_encoded_df], axis=1)
    X_test_final = pd.concat([X_test_set_scaled, X_test_set_encoded_df], axis=1)
    return X_test_final, X_train_final


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let us do a sanity check to see if everything is ok.

    We should have the same rows that we had in the original training. The columns number may have changed due to the encoding and dropping of the "usless_columns".
    """
    )
    return


@app.cell
def _(X_test_final, X_train_final, y_test, y_train):
    # we check the shape of the final training and test sets
    print('The shape of the final training set is:', X_train_final.shape, 'and the shape of y training is:', y_train.shape)
    print('The shape of the final test set is:', X_test_final.shape, 'and the shape of y test is:', y_test.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Conclusion Section 2
    In this section, we learned how to use the Scikit-learn library to prepare the data for a classification task. In particular, we learned how to:
    - Split the data into training and test sets
    - Impute missing values
    - Scale and/or standardize the data
    - Encode categorical features
    - Encode the target (label)

    In the next section, we will use the Scikit-learn library to train and evaluate a machine learning model to predict if a passenger survived the Titanic disaster.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""---""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Training

    In this third (and last) section of this lab, we training and we evaluate our model.
    We will use the Scikit-learn library to train a k-nn model and evaluate its performance. In addition, we will use the Scikit-learn library to find the best hyperparameters for the model and we will print a learning curve to see if the model is overfitting or underfitting.
    Finally, we will compute performance metrics such as the confusion matrix, the accuracy, the precision, the recall and the F1 score.


    **Goals**

    In this part, you will learn how to:

    - Train a k-nn model
    - Evaluate the performance of the model
    - Find the best hyperparameters for the model
    - Print a learning curve
    - Compute performance metrics such as the confusion matrix, the accuracy, the precision, the recall and the F1 score
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**NOTE: this section is dependent on the previous ones, so please make sure you run the previous section before running this one.**""")
    return


@app.cell
def _(X_test_final, X_train_final, y_test, y_train):
    # From the previous notebook, our data is stored in the following variables X_train_final, X_test_final, y_train, y_test
    # to avoid preprocessing the data again and again in the case we do something wrong (and we will), we store a copy of our variables in other variables
    # in case of problems, we can simply re-run this cell to get the pre-processed data back

    # NOTE: the names of the variables below should be the same of the variable at the end of Section 2

    X_tr = X_train_final.copy()
    X_te = X_test_final.copy()
    y_tr = y_train.copy().values.ravel()
    y_te = y_test.copy().values.ravel()


    # Explanation:
    # we use .values.ravel() to convert the pandas dataframe to a one-dimensional numpy array. This is necessary to avoid a warning message when training the model down below.
    # .values will give the values in a numpy array (shape: (n,1), this is actually a 2D array in which the second dimension has size 1)
    # .ravel() will convert that array shape to (n, ) (i.e. flatten it, so it is a 1D array)
    return X_te, X_tr, y_te, y_tr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 3.1 Train a k-nn model (naif way)

    We will use the Scikit-learn library to train a k-nn model. The k-nn algorithm is a non-parametric method used for classification and regression. In this algorithm, the input consists of the k closest training examples in the feature space. The output depends on whether k-nn is used for classification or regression:
    - In [k-nn classification](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small).
    - In [k-nn regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html), the output is the property value for the object. This value is the average of the values of its k nearest neighbors.

    In our case, we will use the k-nn algorithm for classification. In particular to predict if a passenger survived or not.

    In this first "naif" (bad) training, we simply train a model without any tuning. And we test the result in the test set.

    NOTE: this is a "quick & dirty" approach. We will see in the next sections how to properly train and evaluate a model.
    """
    )
    return


@app.cell
def _():
    # Import the libraries
    from sklearn.neighbors import KNeighborsClassifier
    return (KNeighborsClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We will use the Scikit-learn library to train a k-nn model. The k-nn algorithm is a non-parametric method used for classification and regression. In this algorithm, the input consists of the k closest training examples in the feature space. The output depends on whether k-nn is used for classification or regression:
    - In [k-nn classification](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small).
    - In [k-nn regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html), the output is the property value for the object. This value is the average of the values of its k nearest neighbors.

    In our case, we will use the k-nn algorithm for classification. In particular to predict if a passenger survived or not.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's train a k-nn model with the default parameters (k=5, euclidean distance).""")
    return


@app.cell
def _(KNeighborsClassifier, X_tr, y_tr):
    # Train a k-nn model
    knn = KNeighborsClassifier(n_neighbors=5) # n_neighbors is the "k" parameter of k-nn
    knn.fit(X_tr, y_tr)
    return (knn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    After the call to the `.fit()` method, our knn is trained on the training set.

    We can now use the model to predict the labels of the test set.
    """
    )
    return


@app.cell
def _(X_te, knn):
    # Predict the labels of the test set: y_pred
    y_pred = knn.predict(X_te)
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can now evaluate the performance of the model by comparing the prediction to the actual values.""")
    return


@app.cell
def _(y_pred):
    # Evaluate the performance of the model
    print("Test set predictions:\n {}".format(y_pred))
    return


@app.cell
def _(X_te, knn, y_te):
    # Evaluate the performance of the model
    print("Test set accuracy: {:.2f}".format(knn.score(X_te, y_te)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We should have an accuracy of ~0.80, which means that the model is correct 80% of the time on the test set.

    **Questions**:

    - Is considering only "accuracy" enough to evaluate the performance of a model? Why?
    - Are we sure that k=5 is the best value for the hyperparameter k? How can we find the best value for k?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**:

    - No, accuracy is not enough to evaluate the performance of a model. In some cases, accuracy can be misleading. For example, if we have a dataset with 95% of the samples belonging to class 0 and 5% belonging to class 1, a model that always predicts class 0 will have an accuracy of 95%. However, this model is not useful because it does not correctly classify any samples from class 1.
    - No, we are not sure that k=5 is the best value for the hyperparameter k. We can find the best value for k by using cross-validation. Cross-validation is a technique used to evaluate the performance of a model by splitting the dataset into k folds. The model is trained on k-1 folds and tested on the remaining fold. This process is repeated k times, with each fold being used as the test set once. The average performance of the model across all folds is then used to evaluate the model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**NOTE**: this was the "naif" way to train a model. In the next sections, we will see how to properly train and evaluate a model.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 3.2 Train (and fine tune) the model using grid search and cross-validation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In order to find the best `k`, we will use a "*grid search*" approach.

    Grid search is a method for parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid. It is called grid search because it searches across a grid of parameters.
    In our case we have just one hyperparameter: `k`. However, in general we can have more than one hyperparameter. In this case, we will have a "grid" of hyperparameters and Grid search will try *all* the combinations of hyperparameters.
    It is a good approach to find the best hyperparameters for a model but it can be computationally **very expensive** (we are retraining and testing a model for all possible combination of hyperparameters!).
    On the plus side, since each training is independent from the other, grid it can be parallelized easly. 

    The function we will use is the `GridSearchCV` class. The "CV" at the end stands for "Cross Validation". Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The function has a parameter called `cv` that defines the number of folds. It splits the training set into `cv` folds and uses `cv-1` folds for training and the remaining fold for testing. This process is repeated `cv` times, with each of the `cv` folds used exactly once as the validation data. The `GridSearchCV` class will use cross-validation to find the best hyperparameters for the model.

    Wrapping up, the `GridSearchCV` method will train the model for each combination of hyperparameters *cv* times!
    """
    )
    return


@app.cell
def _(KNeighborsClassifier, X_tr, np, y_tr):
    from sklearn.model_selection import GridSearchCV
    _param_grid = {'n_neighbors': np.arange(1, 100)}
    grid_search = GridSearchCV(KNeighborsClassifier(), _param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=1)
    best_knn = grid_search.fit(X_tr, y_tr)
    print('Best parameters: {}'.format(grid_search.best_params_))
    print('Best cross-validation score: {:.2f}'.format(grid_search.best_score_))
    return GridSearchCV, best_knn


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Question**:
    - What is the best `k`? What is the best accuracy score? What does it mean? Is this score related to training or test set?
    - Setting `verbose=1`, you should see something like "Fitting 5 folds for each of 99 candidates, totalling 495 fits". What does it mean?
    - Alternatives of `Grid Search` exist. One of these is `Random Search`. Explain the difference between Grid Search and Random Search. When should you use one over the other?
    - [Optional] Investigate the use of library such as [Optuna](https://optuna.org/) for hyperparameter optimization.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**:

    - Best k is 23. Best accuracy score is 0.82. This score is related to the best hyperparameters found during cross-validation on the training set. It means that the model is correct 82% in average on the validation folds during cross-validation. This score is related to the training set, not the test set.
    - Grid Search vs Random Search:
        - Grid Search: tries all possible combinations of hyperparameters. It is computationally expensive but it guarantees to find the best hyperparameters.
        - Random Search: tries a random combination of hyperparameters. It is less computationally expensive but it does not guarantee to find the best hyperparameters. However, it can be more efficient than Grid Search when the number of hyperparameters is large.

        We should use Grid Search when the number of hyperparameters is small and we want to find the best hyperparameters. We should use Random Search when the number of hyperparameters is large and we want to find a good combination of hyperparameters in a reasonable time.
    - The hyperparameters are chosen according to the result of previous experiments. There is pruning.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 3.2.2 Plot the learning curve to see if the model is overfitting or underfitting

    The learning curve is a plot of the model's performance on the training set and the validation set as a function of the training set size. The purpose of a learning curve is to see how well the model is learning the data. In particular, we want to see if the model is overfitting or underfitting the data.

    In this section, we will plot the learning curve of two models: k-nn with k = 1 (knn_1_neighbor); k-nn with k = 23 (the best_knn we trained above).
    """
    )
    return


@app.cell
def _(KNeighborsClassifier, X_tr, y_tr):
    # Train a k-nn model with k = 1
    knn_1_neighbor = KNeighborsClassifier(n_neighbors=1) # n_neighbors is the "k" parameter of k-nn
    knn_1_neighbor.fit(X_tr, y_tr)
    return (knn_1_neighbor,)


@app.cell
def _(X_tr, best_knn, knn_1_neighbor, np, plt, y_tr):
    from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

    common_params = {
        "X": X_tr,
        "y": y_tr,
        "train_sizes": np.linspace(0.1, 1.0, 5), # 5 different sizes of the training set
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0), # 50 repetitions of 80-20% cross-validation
        "score_type": "both", # we want to see both training and validation scores
        "n_jobs": -1,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }

    for ax_idx, estimator in enumerate([best_knn.best_estimator_, knn_1_neighbor]):
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
        handles, label = ax[ax_idx].get_legend_handles_labels()
        ax[ax_idx].legend(handles[:2], ["Training Score", "Validation Score"])
        ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__} with k={estimator.n_neighbors}")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:

    - What can you say about the learning curve of the model with k = 1? Is the model overfitting or underfitting the data?
    - What can you say about the learning curve of the model with k = 23? Is the model overfitting or underfitting the data?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**:
    #TODO
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Conclusion Section 3
    In this section, we learned how to use the Scikit-learn library to train and fine-tune a machine learning model. In particular, we learned how to use grid search and cross-validation to find the best hyperparameters for the model. We also learned how to plot the learning curve to see if the model is overfitting or underfitting the data.

    In the next section, you will perform the final evaluation of the model using on the **test set**. You will compute performance metrics such as the confusion matrix, the accuracy, the precision, the recall and the F1 score.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4 Evaluation

    We have the best possible model we can fit with the training set (at least in terms of *accuracy*). Now we can evaluate it on the test set. 
    This should be the first time we use the test set. 

    For the evaluation, we will:
    - make the predictions on the test set with the best modelwe have
    - plot the confusion matrix
    - compute the accuracy, the precision, the recall, and the F1 score
    - plot the learning curve (to evaluate possible overfitting or underfitting)
    """
    )
    return


@app.cell
def _(X_te, best_knn):
    y_pred_1 = best_knn.predict(X_te)
    return (y_pred_1,)


@app.cell
def _(y_pred_1, y_te):
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_te, y_pred_1)
    print('Confusion matrix:\n', confusion)
    return confusion, confusion_matrix


@app.cell
def _(confusion, plt, sns):
    # confusion matrix (fancy)

    # we create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    return


@app.cell
def _(confusion, np, plt, sns):
    # confusion matrix (fancy and normalized)
    confusion_normalized = confusion / confusion.sum(axis=1)[:, np.newaxis]

    # we create a heatmap of the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_normalized, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    plt.show()
    return


@app.cell
def _(y_pred_1, y_te):
    from sklearn.metrics import classification_report
    print('Classification report:\n', classification_report(y_te, y_pred_1))
    return (classification_report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:
    - Are you happy with the result? Why?
    - The results that you got on the training set are very different from the results on the test set. What does it mean?
    - What could you do to improve the model?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**:
    - The result are pretty good for such a "naive" model. However, we could do better. The result are better on the majority class (not survived). This is probably due to the fact that the dataset is imbalanced (more not survived than survived).
    - The results are better on the test set than on the training set. This is probably due to the fact that we used cross-validation to find the best hyperparameters. Therefore, the model is not overfitting the training set.
    - We could try to find a way to balance the dataset. We could also try to use a different model. We could also try to use a different set of features. We could also try to use a different set of preprocessing techniques.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Conclusion Section 4
    In this section, we evaluated the performance of a k-nn model on the test set. We computed metrics such as the confusion matrix, the accuracy, the precision, the recall and the F1 score.

    This completes a full Machine Learning workflow. We started from the data exploration, then we prepared the data for the training, we trained and fine-tuned a model, and finally we evaluated the model on the test set. In the next and last section, you will modify the code above to train and evaluate a SVM model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5 Support Vector Machine (SVM)

    In this section you will train and evaluate a Support Vector Machine (SVM) model. We suggest that you directly modify the code above to train, fine-tune, and evaluate a SVM model in addition to the k-nn model. In the evaluation section, you can compare the performance of the two models.

    We suggest using the [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) class from the Scikit-learn library to train a SVM model. The `SVC` class implements the Support Vector Classification algorithm.

    Once implemented and tested, answer the following question below.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Questions**:
    - Which model performs better? (k-nn or SVM)? Why do you think so?
    - Which hyperparameters did you tune for the SVM model? What's the best combination of hyperparameters you found?
    - Which model is faster to train? (k-nn or SVM)? Why do you think so?
    - Discuss the advantages and disadvantages of the two models (k-nn and SVM). Investigate the idea of Lazy learning vs Eager learning.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Answers**:
    #TODO
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Conclusion Section 5

    In this section, you learned how to use the Scikit-learn library to train and evaluate a Support Vector Machine (SVM) model. You also compared the performance of the SVM model with the k-nn model. Now you should have a quite good understanding of the Machine Learning workflow using the Scikit-learn library, in particular for classification tasks.

    #### [Optional] Going further - Regression task
    In this project, we performed a classification task. However, k-nn and SVM can also be used to perform regression tasks.
    We could change the project to a regression task by defining a new target (e.g., the `Age` of the passenger).

    Most of the steps are the same, but there are some important differences. For example, in a regression task:
    - You need to use other metrics such as the mean squared error (MSE) or the mean absolute error (MAE) score.
    - You need to use other classes such as the `KNeighborsRegressor` or the `SVR` class instead of the `KNeighborsClassifier` or the `SVC` class.
    """
    )
    return


@app.cell
def _(
    GridSearchCV,
    X_tr,
    classification_report,
    confusion_matrix,
    np,
    pd,
    plt,
    y_tr,
):
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

    pipe = Pipeline(steps=[('svm', SVC(kernel='rbf', probability=True))])
    _param_grid = {'svm__C': [0.1, 1, 3, 10, 30, 100], 'svm__gamma': ['scale', 'auto', 0.01, 0.03, 0.1, 0.3, 1.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=pipe, param_grid=_param_grid, cv=cv, scoring='f1', n_jobs=-1, refit=True, verbose=1)
    grid.fit(X_tr, y_tr)
    cv_results = pd.DataFrame(grid.cv_results_).sort_values('mean_test_score', ascending=False)
    top10 = cv_results[['params', 'mean_test_score', 'std_test_score', 'mean_fit_time']].head(10)
    print('Top-10 combinaisons (par score CV):')
    print(top10.to_string(index=False))
    print('\nBest params:', grid.best_params_)
    print('Best CV score (moyenne sur folds):', round(grid.best_score_, 4))
    best_model = grid.best_estimator_
    y_hat_tr = best_model.predict(X_tr)
    y_hat_proba_tr = best_model.predict_proba(X_tr)[:, 1]
    print('\nClassification report (sur X_tr, illustratif) :')
    print(classification_report(y_tr, y_hat_tr, digits=3))
    cm = confusion_matrix(y_tr, y_hat_tr)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title('Confusion Matrix (train - illustratif)')
    plt.show()
    acc = accuracy_score(y_tr, y_hat_tr)
    prec = precision_score(y_tr, y_hat_tr, zero_division=0)
    rec = recall_score(y_tr, y_hat_tr)
    f1 = f1_score(y_tr, y_hat_tr)
    try:
        auc = roc_auc_score(y_tr, y_hat_proba_tr)
    except Exception:
        auc = np.nan
    print(f'Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}')
    RocCurveDisplay.from_predictions(y_tr, y_hat_proba_tr)
    plt.title('ROC (train - illustratif)')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
