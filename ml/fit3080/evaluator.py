import json
import os
import re
import subprocess
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


def disclaimer() -> None:
    message = """
    -------------------------------------------------------------------------------
                                    ATTENTION

    Please ensure you are up to date with the latest code changes. Failing to stay 
    updated with the latest code changes puts your work at risk of not being 
    evaluated correctly.
    -------------------------------------------------------------------------------
    I CONFIRM I HAVE PULLED THE LATEST VERSION OF ASSIGNMENT: [y/N]"""

    if input(message).lower() != "y":
        print("", "You MUST!! pull the latest version of the assignment and try again", \
              "This may cause conflicts please accept all incoming changes.", sep="\n")
        exit(1)


def run(command: List[str]) -> subprocess.CompletedProcess:
    """
    Runs a command and returns the completed process.

    Args:
        command (List[str]): The command to run.

    Returns:
        subprocess.CompletedProcess: The completed process.
   """
    try:
        retval = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode('utf-8'))
        exit(1)

    return retval


def read_model_params(file_name: str) -> Dict:
    with open(file_name, 'r', encoding="utf-8") as f:
        contents = f.read()

        regex_search = re.search(r"^#.*({.*})$", contents, re.MULTILINE)
        if regex_search is None:
            print(f"[Model Error] Could not find params for model in {file_name}")
            exit(1)

        param_string = regex_search.group(1)
        param_string = param_string.replace("'", '"')
        try:
            param = json.loads(param_string)
        except json.JSONDecodeError:
            print(f"[Model Error]: params are not in valid format {param_string}")
            exit(1)

        return param


if __name__ == "__main__":

    # disclaimer()

    # Construct Experiments
    question_1a_setup: Dict = {
        'layout': ['./layouts/q1a_bigMaze.lay', './layouts/q1a_bigMaze2.lay', './layouts/q1a_contoursMaze.lay',
                   './layouts/q1a_mediumMaze.lay', './layouts/q1a_mediumMaze2.lay', './layouts/q1a_openMaze.lay',
                   './layouts/q1a_smallMaze.lay', './layouts/q1a_testMaze.lay', './layouts/q1a_tinyMaze.lay',
                   './layouts/q1a_trickyMaze.lay'],
        'model': ['./logs/q1a_bigMaze.model', './logs/q1a_bigMaze2.model', './logs/q1a_contoursMaze.model',
                  './logs/q1a_mediumMaze.model', './logs/q1a_mediumMaze2.model', './logs/q1a_openMaze.model',
                  './logs/q1a_smallMaze.model', './logs/q1a_testMaze.model', './logs/q1a_tinyMaze.model',
                  './logs/q1a_trickyMaze.model'],
        'params': None,
        'num_games': 20,
        'average_score': None,
        'win_rate': None,
    }

    question_1b_setup: Dict = {
        'layout': ['./layouts/q1b_bigMaze.lay', './layouts/q1b_bigMaze2.lay', './layouts/q1b_contoursMaze.lay',
                   './layouts/q1b_mediumMaze.lay', './layouts/q1b_mediumMaze2.lay', './layouts/q1b_openMaze.lay',
                   './layouts/q1b_smallMaze.lay', './layouts/q1b_testMaze.lay', './layouts/q1b_tinyMaze.lay',
                   './layouts/q1b_trickyMaze.lay'],
        'model': ['./logs/q1b_bigMaze.model', './logs/q1b_bigMaze2.model', './logs/q1b_contoursMaze.model',
                  './logs/q1b_mediumMaze.model', './logs/q1b_mediumMaze2.model', './logs/q1b_openMaze.model',
                  './logs/q1b_smallMaze.model', './logs/q1b_testMaze.model', './logs/q1b_tinyMaze.model',
                  './logs/q1b_trickyMaze.model'],
        'params': None,
        'num_games': 20,
        'average_score': None,
        'win_rate': None,
    }

    question_2a_setup: Dict = {
        'layout': ['./layouts/q2a_bigMaze.lay', './layouts/q2a_bigMaze2.lay', './layouts/q2a_contoursMaze.lay',
                   './layouts/q2a_mediumMaze.lay', './layouts/q2a_openMaze.lay', './layouts/q2a_smallMaze.lay',
                   './layouts/q2a_smallMaze2.lay', './layouts/q2a_testMaze.lay', './layouts/q2a_tinyMaze.lay',
                   './layouts/q2a_trickyMaze.lay'],
        'model': ['./logs/q2a_bigMaze.model', './logs/q2a_bigMaze2.model', './logs/q2a_contoursMaze.model',
                  './logs/q2a_mediumMaze.model', './logs/q2a_openMaze.model', './logs/q2a_smallMaze.model',
                  './logs/q2a_smallMaze2.model', './logs/q2a_testMaze.model', './logs/q2a_tinyMaze.model',
                  './logs/q2a_trickyMaze.model'],
        'params': None,
        'score': None
    }

    question_2b_setup: Dict = {
        'layout': ['./layouts/q2b_bigMaze.lay', './layouts/q2b_bigMaze2.lay', './layouts/q2b_contoursMaze.lay',
                   './layouts/q2b_mediumMaze.lay', './layouts/q2b_openMaze.lay', './layouts/q2b_smallMaze.lay',
                   './layouts/q2b_smallMaze2.lay', './layouts/q2b_testMaze.lay', './layouts/q2b_tinyMaze.lay',
                   './layouts/q2b_trickyMaze.lay'],
        'model': ['./logs/q2b_bigMaze.model', './logs/q2b_bigMaze2.model', './logs/q2b_contoursMaze.model',
                  './logs/q2b_mediumMaze.model', './logs/q2b_openMaze.model', './logs/q2b_smallMaze.model',
                  './logs/q2b_smallMaze2.model', './logs/q2b_testMaze.model', './logs/q2b_tinyMaze.model',
                  './logs/q2b_trickyMaze.model'],
        'params': None,
        'score': None
    }

    question_3_setup: Dict = {
        'layout': ['./layouts/q3_mediumClassic.lay', './layouts/q3_openClassic.lay'],
        'model': "logs/q3_weights.model",
        'params': None,
        'num_games': 20,
        'average_score': None,
        'win_rate': None,
    }

    question_1a = pd.DataFrame(question_1a_setup)
    question_1b = pd.DataFrame(question_1b_setup)
    question_2a = pd.DataFrame(question_2a_setup)
    question_2b = pd.DataFrame(question_2b_setup)
    question_3 = pd.DataFrame(question_3_setup)

    # Question 1a
    for index, row in (t := tqdm(question_1a.iterrows(), total=question_1a.shape[0])):
        if not os.path.isfile(row['model']): continue
        if not os.path.isfile(row['layout']): continue

        t.set_description(f"Running Q1a:{row['layout']}")
        question_1a.at[index, 'params'] = read_model_params(row['model'])
        command = ['python', 'pacman.py', '-p', 'Q1Agent', '-a', f"pretrained_values={row['model']}", '-l',
                   row['layout'], '-g', 'StationaryGhost', '-n', str(row['num_games']), '-q']
        result = run(command)

        re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
        question_1a.at[index, 'average_score'] = re_match.group(1) if re_match else None

        re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
        question_1a.at[index, 'win_rate'] = re_match.group(1) if re_match else None

    # Question 1b
    for index, row in (t := tqdm(question_1b.iterrows(), total=question_1b.shape[0])):
        if not os.path.isfile(row['model']): continue
        if not os.path.isfile(row['layout']): continue
        t.set_description(f"Running Q1b:{row['layout']}")
        question_1b.at[index, 'params'] = read_model_params(row['model'])
        command = ['python', 'pacman.py', '-p', 'Q1Agent', '-a', f"pretrained_values={row['model']}", '-l',
                   row['layout'], '-g', 'StationaryGhost', '-n', str(row['num_games']), '-q']
        result = run(command)

        re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
        question_1b.at[index, 'average_score'] = re_match.group(1) if re_match else None

        re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
        question_1b.at[index, 'win_rate'] = re_match.group(1) if re_match else None

    # Question 2a
    for index, row in (t := tqdm(question_2a.iterrows(), total=question_2a.shape[0])):
        if not os.path.isfile(row['model']): continue
        if not os.path.isfile(row['layout']): continue
        t.set_description(f"Running Q2a:{row['layout']}")
        question_2a.at[index, 'params'] = read_model_params(row['model'])
        command = ['python', 'pacman.py', '-p', 'Q2Agent', '-a', f"pretrained_values={row['model']}", '-l',
                   row['layout'], '-g', 'StationaryGhost', '-q']
        result = run(command)

        re_match = re.search(r"Scores:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
        question_2a.at[index, 'score'] = re_match.group(1) if re_match else None

    # Question 2b
    for index, row in (t := tqdm(question_2b.iterrows(), total=question_2b.shape[0])):
        if not os.path.isfile(row['model']): continue
        if not os.path.isfile(row['layout']): continue

        t.set_description(f"Running Q2b:{row['layout']}")
        question_2b.at[index, 'params'] = read_model_params(row['model'])
        command = ['python', 'pacman.py', '-p', 'Q2Agent', '-a', f"pretrained_values={row['model']}", '-l',
                   row['layout'], '-g', 'StationaryGhost', '-q']
        result = run(command)

        re_match = re.search(r"Scores:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
        question_2b.at[index, 'score'] = re_match.group(1) if re_match else None

    # Question 3
    for index, row in (t := tqdm(question_3.iterrows(), total=question_3.shape[0])):
        if not os.path.isfile(row['model']): continue
        if not os.path.isfile(row['layout']): continue

        t.set_description(f"Running Q3:{row['layout']}")
        question_3.at[index, 'params'] = read_model_params(row['model'])
        command = ['python', 'pacman.py', '-p', 'Q3Agent', '-a', f"weights_path={row['model']}", '-l', row['layout'],
                   '-g', 'RandomGhost', '-n', str(row['num_games']), '-q']
        result = run(command)

        re_match = re.search(r"Average\sScore:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
        question_3.at[index, 'average_score'] = re_match.group(1) if re_match else None

        re_match = re.search(r"Win\sRate:\s*(.*)$", result.stdout.decode('utf-8'), re.MULTILINE)
        question_3.at[index, 'win_rate'] = re_match.group(1) if re_match else None

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)

    print("\nEvaluation Report")
    print("=" * 160)
    print(f"Question 1a Results:\n{question_1a.to_markdown()}\n")
    print(f"Question 1b Results:\n{question_1b.to_markdown()}\n")

    print(f"Question 2a Results:\n{question_2a.to_markdown()}\n")
    print(f"Question 2b Results:\n{question_2b.to_markdown()}\n")

    print(f"Question 3 Results:\n{question_3.to_markdown()}\n")
    print("=" * 160)