"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval

This file is a modified version of the humaneval.py file to support the execution of SimPy code.
"""

import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..", '..')))
import time
from spy import Transformer
from evaluate import load

from bigcode_eval.base import Task

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"humaneval_spy": create_task(True)}


def create_task(strip_prompt):
    class HumanEval(GeneralHumanEval):
        def __init__(self):
            super().__init__(strip_prompt)

    return HumanEval


class GeneralHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "openai_humaneval"

    def __init__(self, strip_prompt):
        super().__init__(
            stop_words=["<def_stmt>", "<class_stmt>", "@"],
            requires_execution=True,
        )
        self.strip_prompt = strip_prompt
        self.transformer = Transformer()
      
        

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.strip_prompt:
            decoded = self.transformer.parse(doc['prompt'].strip())
            return decoded[:-11]
        else:
            return doc["prompt"]

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """

        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt):]
        # find the last <block_end> in the generation and remove everything after it
        block_end = generation.rfind("<block_end>")
        if block_end != -1:
            generation = generation[:block_end + len("<block_end>")]
        
        full_generation = prompt + self._stop_at_stop_token(generation, self.stop_words)
        start_time = time.time()
        py_code = self.transformer.decode(full_generation)
        # py_stop_words=["\n#", "\n@", "\nprint", "\nif", "\n```"],
        py_stop_words=['\nif', '\nprint', '<if_stmt>', '\nassert']
        for stop_token in py_stop_words:
            py_code = py_code.split(stop_token)[0]
        self.timer['parse'] += time.time() - start_time
        return py_code

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results
