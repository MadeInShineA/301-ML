# ISC 301, Week 1 Project template
> This repository serves as a template for the projects of ISC 301 week 1.

## Installation 

1. Install [`uv`](https://docs.astral.sh/uv/) on your system 
2. Create Python environment by running 
   ``` 
   uv sync
   ```
3. Install [pre-commit](https://pre-commit.com/):
   ```sh
   uv add install pre-commit
   ```
   Then, install the pre-commit hooks:
   ```sh
   uv run pre-commit install
   ```
4. Try to compile the report template 
   ```
   uv run quarto render reports/presentation_project1/presentation.qmd
   ```

----- 

## Details 
The template includes the following tools 

- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- [`ruff`](https://docs.astral.sh/ruff/) for code formatting.
- [`pyright`](https://github.com/microsoft/pyright) for type checking.
- [pre-commit](https://pre-commit.com/) hooks for automated validation.

### Dependency management

We use [`uv`](https://docs.astral.sh/uv/) for dependency management. It is just as
full-featured as `poetry`, but much faster. 

- Run `uv sync` to synchronize the environment. This will create a virtual environment and install needed development dependencies.
- New dependencies can be installed by running

   ```sh
   uv add polars package-name
   ```

Further references in  `uv`'s [Getting started guide](https://docs.astral.sh/uv/getting-started/).

### Pre-commit hooks

First, install [pre-commit](https://pre-commit.com/):

```sh
uv add install pre-commit
```

Then, install the pre-commit hooks:

```sh
uv run pre-commit install
```

This will create a `.git/hooks/pre-commit` file that will run the pre-commit
hooks every time you commit. Upon the first commit, the hooks will be installed.

Some hooks output error message that require a manual change (e.g., linting
errors). Other hooks perform automated fixes. Either way, you need to re-run
the commit command:

```sh
git commit -m "My message"
```

#### Code formatting

Among the pre-commit hooks, you will find one that runs
[`ruff`](https://docs.astral.sh/ruff/) on every Python file. It is also warmly
recommended that you set up `ruff` in your IDE (e.g., Visual Studio Code, PyCharm).

#### Typing

We recommend the use of [type hints](https://docs.python.org/3/library/typing.html)
of your code. One of the pre-commit hooks is [`pyright`](https://github.com/microsoft/pyright),
which will perform type checking when hints are available. This reduces greatly the
risk of bugs and the maintainability of the code.

### Project Structure 
Inspired from the [cookiecutter data science template](https://github.com/drivendataorg/cookiecutter-data-science).

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Project documentation
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
│
└── {{ src/module_name }}   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes {{ cookiecutter.module_name }} a Python module
    │
    ├── config.py               <- Store useful variables and configuration
```
