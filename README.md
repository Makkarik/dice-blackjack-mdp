# Blackjack, Markov and two dice

## Installation

Before installing the project make sure that you have already installed [poetry](https://pypi.org/project/poetry/) for the dependencies management. 

Also make sure, that your Poetry create the local virtual environment `.venv` in your project's folder.

```shell
poetry config virtualenvs.in-project true
```

After configuring the Poetry, clone the repository and launch the installation.

```shell
git clone git@github.com:Makkarik/dice-blackjack-mdp.git
cd dice-blackjack-mdp
poetry install
```

## Contribution

Keep in mind, that the repository is equipped with pre-commit hooks, so you will be 
unable to push the commit unless all your files satisfy the requirements listed in 
`[tool.ruff.lint]` section in `pyproject.toml` file. 

For better experince, use the VS Code IDE with the installed Ruff extension.
