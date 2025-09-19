# Genetic Algorithms

## Missions & Activities:

<img src="img\Mastermind.jpg" alt="Mastermind" width="300"/>

*Source: [Wikipedia](https://en.wikipedia.org/wiki/Mastermind_(board_game))*

### Activity
- **Problem**: Use a Genetic Algorithm (GA) to find a quasi-optimal solution for a simple problem (i.e., solve a mastermind game).
- **Tools**: [DEAP](https://deap.readthedocs.io/en/stable/index.html) library *or* other libraries (e.g., [PyGAD](https://pygad.readthedocs.io/en/latest/))
- **New glossary**:
  - **Steps**: Initialization, selection, crossover, mutation, evaluation.
  - **Terminology**: Gene, individual (or chromosome), population, parents, children, fitness function, elite.


### Understanding Genetic Algorithms
- Understand the main steps of a Genetic Algorithm (GA). We expect that you are able to **explain** the main steps of a GA:
  - Initialization
  - Selection (explore different selection methods, e.g., tournament selection, roulette wheel selection, etc.)
  - Crossover
  - Mutation
  - Evaluation

- Understand the terminology used in GA. We expect that you are able to **explain** the following terms:
  - Gene
  - Individual (or chromosome)
  - Population
  - Parents
  - Children
  - Fitness function
  - Elite

Note: in this project's folder you will find a simple presentation of the main steps of a GA. We expect that you will explore more in depth the different steps and the terminology using other sources (e.g., online articles, videos, etc.).


### Implementing Genetic Algorithms:

Solve a [mastermind](https://en.wikipedia.org/wiki/Mastermind_(board_game)) game using GA:
- Check the files: 
  - `sentence_mastermind.py` (a simple class providing the game logic: selecting hidden sentence, checking the guess, etc.)
  - `example_mastermind.py` (an example that shows how to call the methods in sentence_mastermind.py)
NOTE: you can use one of the GA libraries mentioned above or implement GA yourself from scratch. The first is faster, the second will require some extra work, but it will provide a better understanding of the implementation details of selection, crossover, mutation, etc. In both cases, you can start from online examples.
- If you use the Deap library, check this [example](https://deap.readthedocs.io/en/stable/examples/ga_onemax.html). Be sure of reading the "stable" version of the doc.
- If you want to code all by yourself, check this [example](https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/).

IMPORTANT: In your Jupyter notebook:
- **Explain** how you implemented the main GA steps in your code: Initialization, selection, crossover, mutation, evaluation.
- **Present** the choice you made for each step (e.g., which selection method you used and why, which crossover method you used and why, how you defined your the fitness function and what's the idea behind it, etc.).
- **Discuss** which "hyperparameters" you tested (e.g., varying number of generations, mutation rate, etc.) and their impact on the results.
- **Discuss** which other hyperparameters you did not test (if any), but would you test if you had more time.

##### Optional task - Solve the same problem using Brute Force! 
Test all possible solutions iteratively! 
Consider the following questions and provide **direct** answers to them in your Jupyter notebook:
- What is the complexity of the Brute Force approach?
- Is it really beneficial to use a GA for this problem?
- How does the time to find the solution compare between the two approaches (Brute Force vs GA)?
- How does it scale compared to the GA? (E.g., compare the results of the GA with the Brute Force approach while increasing the size of the hidden sentence.)

## Expected Outcomes:
To be submitted via GitHub Classroom:
- A **short presentation** (5-10 slides max) of the main steps of a Genetic Algorithm (GA) and how you implemented them in your code (see the [Understanding Genetic Algorithms](#understanding-genetic-algorithms) section). You will prepare this presentation in group 2 people and present it to the class. *The prof. will choose randomly 2 groups to present*.
- A GitHub repository containing:
  - A Jupyter notebook for the optimization of a simple problem.
  - In your notebook, systematically use markdown cells to:
    - Explain the goal of your code.
    - Comment on the results.
    - Provide **direct** answer to the questions in the [Implementing GA](#implementing-ga-provide-these-answers-in-the-jupyter-notebook) section.

## Installation
### uv
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) on your machine. If you want to keep things simple, usually this command is enough:

```bash
pipx install uv
```

- Install the dependencies by running the following command in the root directory of the project (where the `pyproject.toml` file is located):

```bash
uv sync
```
### Without uv
Alternatively, you can install "manually" the dependencies using pip. However, also in this case, we **strongly** suggest to create a virtual environment.

