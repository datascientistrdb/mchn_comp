import os
import time
import argparse
import sys

from allennlp.predictors.predictor import Predictor

model_path = "https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz"


class MachineComprehension:
    def __init__(self, question, answer):
        self._predictor = Predictor.from_path(model_path)
        self._prediction_dict = self.prediction(question, answer)

    def prediction(self, question, answer):
        prediction = self._predictor.predict(question, answer)
        return prediction

    def extract_best_answer(self):
        best_span_str = self._prediction_dict['best_span_str']
        return best_span_str


def file_reader(fname):
    try:
        with open(fname) as freader:
            query = freader.readlines()
        return query[0], query[1]
    except FileNotFoundError as fileEx:
        print(fileEx)
    except IndexError as idxEx:
        print(idxEx)
        print("file is not valid or empty !")
    return None

if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('-f', '--file', help='path of the file', required=True)
    args = cmd_parser.parse_args(sys.argv[1:])
    parsed_args = dict(args._get_kwargs())

    question, answer = file_reader(parsed_args['file'])

    comprehension = MachineComprehension(question, answer)

    print(comprehension.extract_best_answer())