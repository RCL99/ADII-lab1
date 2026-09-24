"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

def count_vowels(data: TextInput) -> int:
    vowels = ['a', 'e', 'i', 'o', 'u']
    text = data.value

    return sum(c in vowels for c in text.lower())


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    already_existing = []

    for c in text:
        if c in already_existing:
            return False
        already_existing.append(c)
    return True


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value

    bits = "{0:b}".format(number)
    return sum(c == '1' for c in bits)


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0

    while number > 9:
        result = 1
        for d in str(number):
            result *= int(d)
        count += 1
        number = result
    return count



def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected

    total = 0

    for prediction, actual in zip(predicted, expected):
        error = prediction - actual
        squared_error = error ** 2
        total += squared_error

    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    div = 2
    result = []

    while number > 1:
        if number % div == 0:
            count = 0
            # count = степень
            while number % div == 0:
                count += 1
                number //= div

            # div ** count
            if count == 1:
                result.append(f"({div})")
            else:
                result.append(f"({div}**{count})")
        else:
            div += 1

    return "".join(result)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 1
    total = 0

    while total < cube_count:
        total += k * k
        if total == cube_count:
            return k
        k += 1

    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    l = len(s)
    mid = l // 2

    if l % 2 != 0:
        left = s[:mid]
        right = s[mid + 1:]
    else:
        left = s[: mid - 1]
        right = s[mid + 1:]

    return sum(int(c) for c in left) == sum(int(c) for c in right)




