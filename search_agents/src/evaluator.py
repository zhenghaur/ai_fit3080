import glob
import os
import subprocess
import sys
from itertools import chain, product
from optparse import OptionParser
from typing import Dict, List


def disclaimer() -> bool:
    message = """
    -------------------------------------------------------------------------------
                                    ATTENTION

    Please ensure you are up to date with the latest code changes. Failing to stay 
    updated with the latest code changes puts your work at risk of not being 
    evaluated correctly.
    -------------------------------------------------------------------------------
    I CONFIRM I HAVE PULLED THE LATEST VERSION OF ASSIGNMENT: [y/N] """

    return input(message)

def linear_product(parameters: Dict) -> List[str]:
    for experiment in product(*parameters.values()):
        yield list(chain(*zip(parameters, experiment)))

def run(all_arguments: Dict):
    print(all_arguments)
    for command in linear_product(all_arguments):
        command.append("-q")
        command.append("-f")
        subprocess.call(command)


if __name__ == "__main__":

    logs_dir = './logs/'
    logs = glob.glob(logs_dir + "*.log")
    for log in logs: os.remove(log)

    layouts_dir = "./layouts/"
    question_1a_pattern = "q1a_*.lay"
    question_1a_layouts = glob.glob(layouts_dir + question_1a_pattern)
    question_1a_args = {
        "python": ["pacman.py"],
        "--pacman": ["SearchAgent"],
        "--agentArgs": ["fn=q1a_solver,prob=q1a_problem"],
        "--layout": question_1a_layouts,
        "--outfile": ["question_1a"],
        "--timeout":["1"]
        }
    
    question_1b_pattern = "q1b_*.lay"
    question_1b_layouts = glob.glob(layouts_dir + question_1b_pattern)
    question_1b_args = {
        "python": ["pacman.py"],
        "--pacman": ["SearchAgent"],
        "--agentArgs": ["fn=q1b_solver,prob=q1b_problem"],
        "--layout": question_1b_layouts,
        "--outfile": ["question_1b"],
        "--timeout":["5"]
        }
    
    question_1c_pattern = "q1c_*.lay"
    question_1c_layouts = glob.glob(layouts_dir + question_1c_pattern)
    question_1c_args = {
        "python": ["pacman.py"],
        "--pacman": ["SearchAgent"],
        "--agentArgs": ["fn=q1c_solver,prob=q1c_problem"],
        "--layout": question_1c_layouts,
        "--outfile": ["question_1c"],
        "--timeout":["30"]
        }
    
    question_2a_patterns = "q2a_*.lay"
    question_2a_layouts = glob.glob(layouts_dir + question_2a_patterns)
    question_2a_args = {
        "python": ["pacman.py"],
        "--pacman": ["Q2A_Agent"],
        "--layout": question_2a_layouts,
        "--outfile": ["question_2a"],
        "--timeout":["30"]
        }
    
    question_2b_patterns = "q2b_*.lay"
    question_2b_layouts = glob.glob(layouts_dir + question_2b_patterns)
    question_2b_args = {
        "python": ["pacman.py"],
        "--pacman": ["Q2B_Agent"],
        "--layout": question_2b_layouts,
        "--outfile": ["question_2b"],
        "--timeout":["30"]
        }
    

    
    if disclaimer() != "y":
        print("")
        exit()


    run(question_1a_args)
    run(question_1b_args)
    run(question_1c_args)
    run(question_2a_args)
    run(question_2b_args)
    



    
