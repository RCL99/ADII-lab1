import unittest
from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)

class TestPythonBasics(unittest.TestCase):
    def test_count_vowels(self):
        self.assertEqual(count_vowels(TextInput("zaza")), 2)
        self.assertEqual(count_vowels(TextInput("zz")), 0)
        self.assertEqual(count_vowels(TextInput("")), 0)

    def test_has_unique_characters(self):
        self.assertTrue(has_unique_characters(TextInput("abcdef")))
        self.assertFalse(has_unique_characters(TextInput("hello")))
        self.assertTrue(has_unique_characters(TextInput("")))

    def test_count_one_bits(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput(1)), 1)   # 1
        self.assertEqual(count_one_bits(PositiveIntegerInput(8)), 1)   # 1000
        self.assertEqual(count_one_bits(PositiveIntegerInput(7)), 3)   # 111

    def test_multiplicative_persistence(self):
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(39)), 3)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(4)), 0)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(999)), 4)

    def test_pyramid(self):
        self.assertEqual(pyramid(PositiveIntegerInput(1)), 1)
        self.assertEqual(pyramid(PositiveIntegerInput(2)), "It is impossible")
        self.assertEqual(pyramid(PositiveIntegerInput(5)), 2)

if __name__ == '__main__':
    unittest.main()
