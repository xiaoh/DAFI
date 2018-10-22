
# standard library imports
import unittest
import os

# local imports
from dainv import utilities as utils

FILE = "test_input.in"


class TestUtilities(unittest.TestCase):

    def setUp(self):
        str_test = 'a = 1'
        str_test += '\n    '
        str_test += '\n   bb 2'
        str_test += '\n   bbb 2'
        str_test += '\n       '
        str_test += '\nc          3 # ignore this!'
        str_test += '\n # d = 4'
        str_test += '\n          e:hello   # a 2'
        str_test += '\nf : world! # end'
        str_test += '\ng : 1,  2, 3,4,   5 #,6, # 7'
        test_file = open(FILE, "w")
        test_file.write(str_test)
        test_file.close()

    def tearDown(self):
        os.remove(FILE)

    def test_utils(self):
        pattern = 'bb 2'
        subst = 'b 2'
        utils.replace(FILE, pattern, subst)
        input_dict = utils.read_input_data(FILE)
        input_dict['g'] = utils.extract_list_from_string(
            input_dict['g'], sep=',', type_func=int)
        input_dict_correct = {'a': '1',
                              'b': '2',
                              'bb': '2',
                              'c': '3',
                              'e': 'hello',
                              'f': 'world!',
                              'g': [int(1), int(2), int(3), int(4), int(5)]
                              }

        self.assertEqual(input_dict, input_dict_correct)


if __name__ == '__main__':
    unittest.main()
