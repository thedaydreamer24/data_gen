import math
import json
import random
import re
from fractions import Fraction as OldFraction
# from .parser_custom_functions.lesson_time import *

class Fraction(OldFraction):
    def __new__(cls, numerator, denominator=1):
        # Use integers directly if denominator is 1, otherwise create a fraction
        # if denominator == 1 and isinstance(numerator, int):
        #     return numerator
        obj = super().__new__(cls, numerator, denominator)
        obj.original_numerator = numerator
        obj.original_denominator = denominator
        return obj

    def __str__(self):
        return f"{self.original_numerator}/{self.original_denominator}" if self.original_denominator != 1 else f"{self.original_numerator}"
    
    def __repr__(self):
        return self.__str__()
    
    def __int__(self):
        # This converts the fraction to an integer
        return int(self.original_numerator / self.original_denominator)
    
    
# Custom JSON encoder for Fraction
class FractionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Fraction):
            return str(obj)
        return super().default(obj)

# Custom JSON decoder for Fraction
def fraction_decoder(dct):
    if '__fraction__' in dct:
        return Fraction(dct['numerator'], dct['denominator'])
    return dct
    
def custom_fraction(numerator, denominator=1):
    """Create a fraction without reducing it by manipulating the raw values."""
    if isinstance(numerator, Fraction) and isinstance(denominator, Fraction):
        return Fraction(
            numerator.original_numerator * denominator.original_denominator,
            numerator.original_denominator * denominator.original_numerator
        )
    elif isinstance(numerator, Fraction):
        return Fraction(numerator.original_numerator, numerator.original_denominator * denominator)
    elif isinstance(denominator, Fraction):
        return Fraction(numerator * denominator.original_denominator, denominator.original_numerator)
    else:
        return Fraction(numerator, denominator)


def preprocess_expression(expression, variables):
    if not variables:
        return expression
    if expression.isdigit():
        return expression
    # print("EXPRESSION", expression, flush=True)
    # Add '*' between a number or variable and an opening parenthesis
    expression = re.sub(r"(\d+|\w)(\()", r"\1*(", expression)
    # Add '*' between a closing and an opening parenthesis
    expression = re.sub(r"(\))(\()", r"\1*(", expression)
    # Add '*' between a closing parenthesis and a variable or number
    expression = re.sub(r"(\))(\d+|\w)", r"\1*\2", expression)
    # Add '*' between consecutive variables and numbers, e.g., `3x` or `xy`
    expression = re.sub(r"(\d+)([" + "".join(variables) + "])", r"\1*\2", expression)
    expression = re.sub(r"([" + "".join(variables) + "])(\d+)", r"\1*\2", expression)
    # Handle variable to variable multiplication like xy becoming x*y
    expression = re.sub(r"([" + "".join(variables) + "])([" + "".join(variables) + "])", r"\1*\2", expression)
    # print("PROCESSED", expression, flush=True)
    return expression

# def preprocess_expression(expression, variables):
#     # Correct handling of variables to avoid regex errors due to empty character sets or malformed expressions.
#     if variables:
#         var_pattern = "[" + "".join(re.escape(var) for var in variables) + "]"
#     else:
#         var_pattern = r"\w"  # Default to word characters if no specific variables are defined

#     # Add '*' between a number or variable and an opening parenthesis
#     expression = re.sub(r"(\d+|\w)(\()", r"\1*(", expression)
#     # Add '*' between a closing and an opening parenthesis
#     expression = re.sub(r"(\))(\()", r"\1*(", expression)
#     # Add '*' between a closing parenthesis and a variable or number
#     expression = re.sub(r"(\))(\d+|\w)", r"\1*\2", expression)
#     # Add '*' between consecutive variables and numbers, e.g., `3x` or `xy`
#     expression = re.sub(r"(\d+)(" + var_pattern + ")", r"\1*\2", expression)
#     expression = re.sub(r"(" + var_pattern + ")(\d+)", r"\1*\2", expression)
#     # Handle variable to variable multiplication like xy becoming x*y
#     expression = re.sub(r"(" + var_pattern + ")(" + var_pattern + ")", r"\1*\2", expression)
#     return expression



# def evaluate_expression(expression, context):
#     expression = preprocess_expression(expression, context.keys())
#     if '/' in expression:
#         numerator, denominator = expression.split('/')
#         num_val = eval(numerator, {"__builtins__": None, "Fraction": Fraction}, context)
#         den_val = eval(denominator, {"__builtins__": None, "Fraction": Fraction}, context)
#         return Fraction(num_val, den_val)  # Create a fraction if division is involved
#     # Evaluate expression directly otherwise
#     return eval(expression, {"__builtins__": None, "int": int, "Fraction": Fraction}, context)


recognized_keywords = {
    'unique_options',
    'answer',
    'order',
    'options_length',
    'options_type',
    'carry_over',
    'question',
    'incorrect_length',
    'borrow',
    'carry_over_expr',
    'borrow_expr',
    'compare',
    'correct_equivalent',
    'show_labels',
    'hide_integers',
    'show_endpoints',
    'shape_merge',
    'time_delta',
}

skip_preprocessing = {
    'incorrect_1',
    'incorrect_2',
}


def evaluate_expression(expression, context):
    # Preprocess to handle implicit multiplication
    # print("EXPRESSION", expression, flush=True)
    # print("CONTEXT", context, flush=True)
    if any(keyword in expression for keyword in skip_preprocessing):
        processed_expression = expression
    else:
        processed_expression = preprocess_expression(expression, context.keys())

    # processed_expression = preprocess_expression(expression, context.keys())
    # print("PROCESSED", processed_expression, flush=True)
    if '/' in processed_expression:
        # Carefully split the expression into numerator and denominator
        parts = processed_expression.split('/')
        numerator = '/'.join(parts[:-1])
        denominator = parts[-1]
        num_val = eval(numerator, {"__builtins__": None, "Fraction": Fraction}, context)
        den_val = eval(denominator, {"__builtins__": None, "Fraction": Fraction}, context)
        return custom_fraction(num_val, den_val)
    return eval(processed_expression, {"__builtins__": None, "int": int, "Fraction": Fraction}, context)


def parse_range(expression, context):
    # print("Range expression", expression, flush=True)
    args = expression.strip().replace('range(', '').replace(')', '').split(',')
    # print("Range args", args, flush=True)
    for i in range(len(args)):
        if args[i].isdigit():
            args[i] = int(args[i])
        else:
            args[i] = evaluate_expression(args[i], context)
            if isinstance(args[i], Fraction):
                args[i] = int(args[i])
    if len(args) == 2:
        # print("ARGS", args, flush=True)
        return lambda: random.randint(*args)
    elif len(args) == 3:
        # print("ARGS", args, flush=True)
        return lambda: random.randrange(*args)
    
# def parse_range(expression, context):
#     args = expression.strip('range()').split(',')
#     args = [evaluate_expression(arg, context) for arg in args]
#     if len(args) == 3:
#         start, end, step = args
#         return lambda: random.randrange(start, end, step)
#     elif len(args) == 2:
#         start, end = args
#         return lambda: random.randint(start, end)

# def closest_sum(d, e):
#     # d: number of digits in both numbers
#     # e: estimate to the multiple of e
#     # n: estimation value which is multiple of e
#     # x: number 1
#     # y: number 2

#    # pick n which is multiple of e between 2*d, 2*(10^d)-2

#     n = e * round(random.randint(2*10**(d-1), 2*10**d-2)/e)
#     sum = n + random.randint((-e//2)+1, (e//2)-1)
#     x = random.randint(10^(d-1), sum-12)
#     y = sum - x

#     return n , x, y

def estimate(n):
    if n == 0:
        return 0

    # Determine the order of magnitude of the number
    magnitude = 10 ** int(math.log10(abs(n)))

    # Find the lower multiple
    lower_multiple = (n // magnitude) * magnitude

    # Find the higher multiple
    higher_multiple = lower_multiple + magnitude

    # Find the midpoint between the lower and higher multiples
    midpoint = (lower_multiple + higher_multiple) / 2

    # Return the closer multiple based on the midpoint
    if n < midpoint:
        return lower_multiple
    else:
        return higher_multiple


def closest_sum(d, e):
    # Generate a random sum n that is a multiple of e within specified limits
    lower_bound = 2 * 10**(d-1)
    upper_bound = 2 * 10**d - 2
    n = e * round(random.randint(lower_bound, upper_bound) / e)

    # Adjust n to be close to some number within ±e/2
    adjustment = random.randint(-e//2 + 1, e//2 - 1)
    sum_close = n + adjustment

    # Generate x and ensure it's within the valid range
    if 10**(d-1) < sum_close - 10**(d-1):
        x = random.randint(10**(d-1), sum_close - 10**(d-1))
    else:
        # Adjust if the range for x is invalid
        x = 10**(d-1)
    y = sum_close - x

    # if number of digits in x or y is not equal to d, then generate new x and y
    if len(str(x)) != d or len(str(y)) != d:
        print("RETRYING", x, y, flush=True)
        return closest_sum(d, e)
    
    if estimate(x) + estimate(y) != estimate(n):
        print("RETRYING", x, y, flush=True)
        return closest_sum(d, e)

    return n, x, y

def abs_value(x):
    """Returns the absolute value of the given number."""
    if isinstance(x, (int, float)):
        return abs(x)
    else:
        raise TypeError("Unsupported type for abs_value. Must be int or float.")

def closest_diff(d, e):
    max_val = 10**d - 10**(d-1) - 1
    min_val = 10**(d-1)

    n = e * random.randint((min_val // e) + 1, ((max_val-2*min_val) // e) - 1)
    adjustment = random.randint(max(-e//2 + 1,-min_val), min(e//2 - 1, min_val)) 
    diff = n + adjustment
    print("DIFF", diff, n, adjustment, min_val, max_val, flush=True)
    if diff<min_val:
        x = random.randint(min_val, max_val)
        y = x - diff
    else:
        y = random.randint(min_val, max_val-diff)
        x = y + diff

     # if number of digits in x or y is not equal to d, then generate new x and y
    if len(str(x)) != d or len(str(y)) != d:
        print("RETRYING", x, y, flush=True)
        return closest_sum(d, e)

    return n, x, y

def closest_diff2(d, e):
    max_val = 10**d - 1
    min_val = 10**(d-1)

    # Redefine n to span more widely and still respect the `e` multiple constraint
    n = random.randint(min_val // e, max_val // e) * e

    # Simpler random adjustment, ensuring it stays within bounds
    adjustment = random.randint((-e//2)+1, (e//2)-1)
    
    diff = n + adjustment
    print("DIFF", diff, n, adjustment, min_val, max_val, flush=True)

    # Ensure diff does not take y out of the [min_val, max_val] range
    if diff < min_val:
        x = random.randint(min_val, max_val)
        y = x - diff
    else:
        max_y = max_val - diff
        if max_y < min_val:
            return closest_diff2(d, e)  # Retry if y would be out of bounds
        y = random.randint(min_val, max_y)
        x = y + diff

    # Check for digit count and retry if necessary
    if len(str(x)) != d or len(str(y)) != d:
        print("RETRYING", x, y, flush=True)
        return closest_diff2(d, e)

    if estimate(x) - estimate(y) != estimate(diff):
        print("RETRYING", x, y, flush=True)
        return closest_diff2(d, e)

    return n, x, y

def divisor(y):
    divisors = [i for i in range(2, y) if y % i == 0]
    return random.choice(divisors) if divisors else 1

def smallest_divisor(n):
    """Returns the smallest divisor of n greater than 1, or n if n is prime."""
    if n <= 1:
        return n
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i
    return n  # n is prime

def rev(x,y=None):
    if y:
        join = str(x) + str(y)
        return int(join[::-1])
    if isinstance(x, int):
        return int(str(x)[::-1])
    return x

def subtract_without_borrow(x, y):
    # do the subtraction without borrowing
    # x = abcd, y = efgh, result = ijkl, where i = a-e, j = b-f, k = c-g, l = d-h , if i< 0, i = i+10
    x = str(x)
    y = str(y)
    # if y is shorter than x, add leading zeros to y
    if len(y) < len(x):
        y = '0'*(len(x)-len(y)) + y
    result = []
    for i in range(len(x)):
        result.append(int(x[i]) - int(y[i]))
    for i in range(len(result)):
        if result[i] < 0:
            result[i] += 10
    return int(''.join(map(str, result)))

def no_carry_sum_range(min_a, max_a, min_b, max_b, min_sum=11, max_sum=100):
    # Generate list of pairs (a, b) where a + b has no carry
    pairs = [(a, b) for a in range(min_a, max_a + 1)
                    for b in range(min_b, max_b + 1) if not has_carry(a, b)]
    # Filter pairs to only include those whose sum is within the specified range
    pairs = [(a, b) for a, b in pairs if min_sum <= a + b <= max_sum]
    if pairs:
        result = random.choice(pairs)
        # randomly swap the pair
        if random.choice([True, False]):
            return result[1], result[0]
        return result
    else:
        return None

def has_carry(a, b):
    # Check if there's a carry in any digit
    while a > 0 and b > 0:
        if (a % 10 + b % 10) >= 10:
            return True
        a //= 10
        b //= 10
    return False

def no_borrow_diff_range(min_p, max_p, min_q, max_q, min_diff=0, max_diff=100):
    if min_p < min_q and max_p < max_q:
        #swap
        min_p, min_q = min_q, min_p
        max_p, max_q = max_q, max_p
    # Generate list of pairs (p, q) where p - q has no borrow
    pairs = [(p, q) for p in range(min_p, max_p + 1)
                    for q in range(min_q, min(p + 1, max_q + 1)) if not has_borrow(p, q)]
    pairs = [(p, q) for p, q in pairs if min_diff <= p - q <= max_diff]
    # filter out pairs with same values
    pairs = [(p, q) for p, q in pairs if p > q]
    if pairs:
        return random.choice(pairs)
    else:
        return None

def has_borrow(p, q):
    # Check if there's a borrow in any digit
    while p > 0 and q > 0:
        if (p % 10) < (q % 10):
            return True
        p //= 10
        q //= 10
    return False

def must_carry_sum_range(min_a, max_a, min_b, max_b, min_sum=11, max_sum=100):
    # Generate list of pairs (a, b) where a + b must have a carry
    pairs = [(a, b) for a in range(min_a, max_a + 1)
                    for b in range(min_b, max_b + 1) if has_carry(a, b)]
    # Filter pairs to only include those whose sum is within the specified range
    pairs = [(a, b) for a, b in pairs if min_sum <= a + b <= max_sum]
    if pairs:
        result = random.choice(pairs)
        # randomly swap the pair
        if random.choice([True, False]):
            return result[1], result[0]
        return result
    else:
        return None

def must_borrow_diff_range(min_p, max_p, min_q, max_q, min_diff=0, max_diff=100):
    if min_p < min_q and max_p < max_q:
        #swap
        min_p, min_q = min_q, min_p
        max_p, max_q = max_q, max_p
    # Generate list of pairs (p, q) where p - q must have a borrow
    pairs = [(p, q) for p in range(min_p, max_p + 1)
                    for q in range(min_q, max_q + 1) if has_borrow(p, q)]
    pairs = [(p, q) for p, q in pairs if min_diff <= p - q <= max_diff]
    # filter out pairs with same values
    pairs = [(p, q) for p, q in pairs if p > q]
    if pairs:
        return random.choice(pairs)
    else:
        return None


def has_no_carry_over_mul(a, b):
    while a >= 10:  
        if (a % 10) * b >= 10:
            return False
        a //= 10  
    return True  

def no_carry_over_mul_range(min_x, max_x, min_y, max_y):
    # Generate list of pairs (x, y) where x * y has no carry over in all but the highest place digit
    pairs = [(x, y) for x in range(min_x, max_x + 1)
                    for y in range(min_y, max_y + 1) if has_no_carry_over_mul(x, y)]
    if pairs:
        result = random.choice(pairs)
        # randomly swap the pair
        if random.choice([True, False]):
            return result[1], result[0]
        return result
    else:
        return None

def has_at_least_one_carry_over_mul_helper(x, y):
    # Initialize a carry variable
    carry = 0
    digit_position = 1
    digits_in_x = len(str(x))
    
    # Iterate through each digit of x
    while x > 0:
        # Get the last digit of x
        last_digit_x = x % 10
        x //= 10
        
        # Calculate the product of the last digit of x and y, plus any existing carry
        product = (last_digit_x * y) + carry
        
        # Determine the new carry
        carry = product // 10
        
        # Check if there was a carry from this digit multiplication
        if carry > 0 and digit_position < digits_in_x:
            return True
        
        digit_position +=1
    
    # If we finish the loop and found no carry over at any digit multiplication, return False
    return False

def has_at_least_one_carry_over_mul(x,y):
    #swap x and y if x is smaller than y
    if x < y:
        x, y = y, x
    
    while y > 0:
        last_digit_y = y % 10
        y //= 10
        
        if has_at_least_one_carry_over_mul_helper(x, last_digit_y):
            return True
    
    # If we finish the loop and found no carry over at any digit multiplication, return False
    return False


def with_carry_over_mul_range(min_x, max_x, min_y, max_y):
    # Generate list of pairs (x, y) where at least one digit of x * y must result in carry over
    pairs = [(x, y) for x in range(min_x, max_x + 1)
                    for y in range(min_y, max_y + 1) if has_at_least_one_carry_over_mul(x, y)]
    if pairs:
        result = random.choice(pairs)
        # randomly swap the pair
        if random.choice([True, False]):
            return result[1], result[0]
        return result
    else:
        return None
    
def subtract_tenth_place(p, q):
    """
    Subtracts a one-digit number from a two-digit number by subtracting the one-digit number from the
    tens place of the two-digit number. If both numbers are two-digit, it returns their simple difference.
    """
    # Case where p is a two-digit number and q is a one-digit number
    if 10 <= abs(p) < 100 and -10 < q < 10:
        result = p - 10 * q
    # Case where q is a two-digit number and p is a one-digit number
    elif 10 <= abs(q) < 100 and -10 < p < 10:
        result = q - 10 * p
    # If both are two-digit numbers, return their simple difference
    elif 10 <= abs(p) < 100 and 10 <= abs(q) < 100:
        result = p - q
    # If neither is a two-digit number or both are one-digit, return the simple difference
    else:
        result = p - q

    return result

    
def add_tenth_place(p, q):
    """
    Adds a two-digit number with a one-digit number by adding the one-digit number to the tens place
    of the two-digit number. If both numbers are two-digit, it returns their simple sum.
    """
    # Case where p is a two-digit number and q is a one-digit number
    if 10 <= abs(p) < 100 and -10 < q < 10:
        result = p + 10 * q
    # Case where q is a two-digit number and p is a one-digit number
    elif 10 <= abs(q) < 100 and -10 < p < 10:
        result = q + 10 * p
    # If both are two-digit numbers, return their simple sum
    elif 10 <= abs(p) < 100 and 10 <= abs(q) < 100:
        result = p + q
    # If neither is a two-digit number or both are one-digit, return the simple sum
    else:
        result = p + q

    return result


def add_without_carry_over(p, q):
    """
    Adds two numbers p and q ignoring the carry-over digits.
    If there is no carry-over, simply returns p + q.
    Handles negative numbers as well.
    """
    # Extract ones and tens digits, handle negative numbers
    p_ones = abs(p) % 10
    p_tens = (abs(p) // 10) % 10
    q_ones = abs(q) % 10
    q_tens = (abs(q) // 10) % 10

    # Calculate sum of ones digits and tens digits separately
    sum_ones = p_ones + q_ones
    sum_tens = p_tens + q_tens

    # Check if there is any carry-over
    if sum_ones >= 10 or sum_tens >= 10:
        # If sum of ones digits exceeds 9, ignore carry-over
        if sum_ones >= 10:
            sum_ones -= 10
        # If sum of tens digits exceeds 9, ignore carry-over
        if sum_tens >= 10:
            sum_tens -= 10
        # Combine the results
        result = sum_tens * 10 + sum_ones
        # Adjust for the sign of the result
        result = -result if p < 0 or q < 0 else result
    else:
        # If there is no carry-over, return the simple sum
        result = p + q

    return result

def subtract_without_borrow(p, q):
    """
    Subtracts q from p. If borrowing would be needed (i.e., the ones place of p is less than the ones place of q
    when p > q), it subtracts an additional 10 to account for the borrowing. If p < q, it subtracts an additional
    10 from the result. If no borrowing is needed, it simply returns p - q.
    """
    # Calculate the ones place of p and q
    p_ones = p % 10
    q_ones = q % 10
    
    # Perform the subtraction
    result = p - q
    
    # Check if borrowing would be needed and adjust the result accordingly
    if p > q and p_ones < q_ones:
        result -= 10
    elif p < q:
        result -= 10
    
    return result

def generate_sum_questions(is_carry, min_n, max_n, min_a, max_a, min_b, max_b, num_options=4):
    attempts = 0
    while attempts < 10000:  # Limit the number of retries to prevent infinite loops
        N = random.randint(min_n, max_n)
        equal_probability = 1/num_options
        probabilities = [equal_probability] * num_options
        num_correct = random.choices([1, 2, 3, 4], weights=probabilities)[0]
        options = []
        correct_answers = []
        if not is_carry:
            number_of_carry_overs = 0
        else:
            number_of_carry_overs = random.randint(1, num_options)

        # Generate correct options
        for _ in range(num_correct):
            if number_of_carry_overs > 0:
                pair = must_carry_sum_range(min_a, min(max_a, N - 1), min_b, max_b, N, N)
                if pair is None:
                    continue
                a, b = pair
                number_of_carry_overs -= 1
            else:
                pair = no_carry_sum_range(min_a, min(max_a, N - 1), min_b, max_b, N, N)
                if pair is None:
                    continue  # Try to find another valid N
                a, b = pair

            if min_a <= a <= max_a and min_b <= b <= max_b:
                option = f"{a} + {b}"
                if option in options:
                    continue # Avoid duplicate options
                options.append(option)
                correct_answers.append(option)

        # Generate incorrect options
        for _ in range(num_options - num_correct):
            # find the intersection of the ranges n-10, n+10 and min_n, max_n
            incorrect_N = random.randint(max(min_n, N-10), min(max_n, N+10))
            while incorrect_N == N:  # Ensure incorrect options do not sum to N
                incorrect_N = random.randint(max(min_n, N-10), min(max_n, N+10))

            if number_of_carry_overs > 0:
                pair = must_carry_sum_range(min_a, min(max_a, incorrect_N - 1), min_b, max_b, incorrect_N, incorrect_N)
                if pair is None:
                    continue
                a, b = pair
                number_of_carry_overs -= 1
            else:
                pair = no_carry_sum_range(min_a, min(max_a, incorrect_N - 1), min_b, max_b, incorrect_N, incorrect_N)
                if pair is None:
                    continue  # Try to find another valid incorrect_N
                a, b = pair

            if min_a <= a <= max_a and min_b <= b <= max_b:
                if f"{a} + {b}" in options:
                    continue # Avoid duplicate options
                options.append(f"{a} + {b}")

        if len(options) == num_options:
            return {
                "sum": N,
                "options": options,
                "correct_answers": correct_answers
            }
        attempts += 1
    return None  # If no valid configuration is found after 100 attempts


def generate_sum_questions2(is_carry, min_n, max_n, min_a, max_a, min_b, max_b, num_options=4):
    attempts = 0
    while attempts < 100:  # Limit the number of retries to prevent infinite loops
        N = random.randint(min_n, max_n)
        equal_probability = 1/num_options
        probabilities = [equal_probability] * num_options
        num_correct = random.choices([1, 2, 3, 4], weights=probabilities)[0]
        options = []
        correct_answers = []
        number_of_carry_overs = random.randint(1, num_options) if is_carry else 0

        # Generate correct and incorrect options
        for i in range(num_options):
            current_is_correct = i < num_correct
            current_N = N if current_is_correct else random.randint(min_n, max_n)

            if number_of_carry_overs > 0 or current_is_correct:
                a_range_min = min_a
                a_range_max = min(max_a, current_N - 1)
                if a_range_min <= a_range_max:
                    a = random.randint(a_range_min, a_range_max)
                    b = current_N - a
                    option = f"{a} + {b}"
                    if current_is_correct:
                        correct_answers.append(option)
                    options.append(option)
                number_of_carry_overs -= 1
            else:
                # Ensure at least one incorrect option
                incorrect_a = random.randint(min_a, max_a)
                incorrect_b = random.randint(min_b, max_b)
                if incorrect_a + incorrect_b != N:
                    options.append(f"{incorrect_a} + {incorrect_b}")

        if len(options) == num_options:
            return {
                "sum": N,
                "options": options,
                "correct_answers": correct_answers
            }
        attempts += 1
    return None  # If no valid configuration is found after 100 attempts


def generate_diff_questions(is_borrow, min_d, max_d, min_a, max_a, min_b, max_b, num_options=4):
    attempts = 0
    while attempts < 10000:  # Limit the number of retries to prevent infinite loops
        D = random.randint(min_d, max_d)
        equal_probability = 1/num_options
        probabilities = [equal_probability] * num_options
        num_correct = random.choices([1, 2, 3, 4], weights=probabilities)[0]
        options = []
        correct_answers = []
        if not is_borrow:
            number_of_borrows = 0
        else:
            number_of_borrows = random.randint(1, num_options)

        print("Number_of_borrows", number_of_borrows, flush=True)
        print("Num_correct", num_correct, flush=True)

        # Generate correct options
        for _ in range(num_correct):
            if number_of_borrows > 0:
                pair = must_borrow_diff_range(min_a, max_a, min_b, max_b, D, D)
                if pair is None:
                    break
                a, b = pair
                number_of_borrows -= 1
            else:
                pair = no_borrow_diff_range(min_a, max_a, min_b, max_b, D, D)
                if pair is None:
                    break  
                a, b = pair
            if pair is None:
                continue # Try to find another valid D

            if min_a <= a <= max_a and min_b <= b <= max_b and a > b:
                option = f"{a} - {b}"
                if option in options:
                    continue # Avoid duplicate options
                options.append(option)
                correct_answers.append(option)

        # Generate incorrect options
        for _ in range(num_options - num_correct):
            incorrect_D = random.randint(max(min_d, D - 10), min(max_d, D + 10))
            while incorrect_D == D:  # Ensure incorrect options do not result in D
                incorrect_D = random.randint(max(min_d, D - 10), min(max_d, D + 10))

            if number_of_borrows > 0:
                pair = must_borrow_diff_range(min_a, max_a, min_b, max_b, incorrect_D, incorrect_D)
                if pair is None:
                    break
                a, b = pair
                number_of_borrows -= 1
            else:
                pair = no_borrow_diff_range(min_a, max_a, min_b, max_b, incorrect_D, incorrect_D)
                if pair is None:
                    break
                a, b = pair
            if pair is None:
                continue

            if min_a <= a <= max_a and min_b <= b <= max_b and a > b:
                if f"{a} - {b}" in options:
                    continue # Avoid duplicate options
                options.append(f"{a} - {b}")

        if len(options) == num_options:
            return {
                "diff": D,
                "options": options,
                "correct_answers": correct_answers
            }
        attempts += 1
    return None  # If no valid configuration is found after 100 attempts

def generate_mul_questions(is_borrow, min_x, max_x, min_y, max_y, num_options=4):
    attempts = 0
    while attempts < 10000:  # Limit the number of retries to prevent infinite loops
        num_correct = random.choices([1, 2], weights=[0.5, 0.5])[0]  # Choose between 1 or 2 correct options
        options = []
        correct_answers = []
        X = None

        # Generate correct options
        for _ in range(num_correct):
            while True:
                a = random.randint(min_x, max_x)
                b = random.randint(min_y, max_y)
                if is_borrow:
                    if has_at_least_one_carry_over_mul(a, b):
                        if X is None:
                            X = a * b
                            break
                        elif a * b == X:
                            break
                else:
                    if has_no_carry_over_mul(a, b):
                        if X is None:
                            X = a * b
                            break
                        elif a * b == X:
                            break
            
            if min_x <= a <= max_x and min_y <= b <= max_y:
                option = f"{a} × {b}"
                if option not in options:
                    options.append(option)
                    correct_answers.append(option)

        if not correct_answers:  # If no valid correct options were found, skip to the next attempt
            attempts += 1
            continue

        # Generate incorrect options
        for _ in range(num_options - num_correct):
            while True:
                a = random.randint(min_x, max_x)
                b = random.randint(min_y, max_y)
                incorrect_X = a * b
                if incorrect_X != X:
                    if is_borrow:
                        if has_at_least_one_carry_over_mul(a, b):
                            break
                    else:
                        if has_no_carry_over_mul(a, b):
                            break

            if min_x <= a <= max_x and min_y <= b <= max_y:
                option = f"{a} × {b}"
                if option not in options:
                    options.append(option)

        if len(options) == num_options:
            return {
                "product": X,
                "options": options,
                "correct_answers": correct_answers
            }
        attempts += 1
    return None


def generate_mul_options2(min_x, max_x, digits_y, num_options, unique=False):
    attempts = 0
    options = []
    used_products = set()
    used_a_values = set()

    while attempts < 10000:  # Limit the number of retries to prevent infinite loops
        while len(options) < num_options:
            a = random.randint(min_x, max_x)
            b = random.choice(digits_y)  # Pick a random digit from digits_y
            option = f"{a} × {b}"
            product = a * b

            if unique and (product in used_products or a in used_a_values):
                continue  # Skip if product or a is already used and unique is required

            if option not in options:
                options.append(option)
                if unique:
                    used_products.add(product)
                    used_a_values.add(a)

        if len(options) == num_options:
            return {
                "options": options
            }
        attempts += 1
        options = []  # Reset options for next attempt
        used_products.clear()  # Reset used products for next attempt if unique is required
        used_a_values.clear()  # Reset used a values for next attempt if unique is required

    return {
        "options": options
    }

def generate_div_quotient(min_a, max_a, num_options, divisors):
    # Determine the maximum number of correct options based on the number of divisors
    max_correct_options = min(len(divisors), 4)
    num_correct = random.choices(range(1, max_correct_options + 1), k=1)[0]

    options = []
    correct_answers = set()
    used_dividends = set()
    quotient = None

    divisor_quotient_map = {}

    # Create a map of quotients to potential dividend and divisor pairs
    for divisor in divisors:
        for a in range(min_a, max_a + 1):
            if a % divisor == 0:
                q = a // divisor
                if q not in divisor_quotient_map:
                    divisor_quotient_map[q] = []
                divisor_quotient_map[q].append((a, divisor))

    while len(correct_answers) < num_correct:
        if not divisor_quotient_map:
            return None
        quotient = random.choice(list(divisor_quotient_map.keys()))
        if len(divisor_quotient_map[quotient]) >= num_correct:
            selected_pairs = random.sample(divisor_quotient_map[quotient], num_correct)
            valid_selection = True
            for a, b in selected_pairs:
                if a in used_dividends:
                    valid_selection = False
                    break
            if valid_selection:
                for a, b in selected_pairs:
                    correct_answers.add(f"{a} ÷ {b}")
                    options.append(f"{a} ÷ {b}")
                    used_dividends.add(a)
                break

    # Ensure we have correct answers before generating incorrect options
    if not correct_answers:
        return None

    # Track the usage of each divisor
    used_divisors = [int(pair.split(' ÷ ')[1]) for pair in correct_answers]
    remaining_divisors = [d for d in divisors if d not in used_divisors]

    # Distribute divisors for incorrect options to ensure even distribution
    divisor_usage = {div: 0 for div in divisors}
    for div in used_divisors:
        divisor_usage[div] += 1

    attempts = 0
    while len(options) < num_options and attempts < 10000:
        a = random.randint(min_a, max_a)
        if remaining_divisors:
            b = remaining_divisors[len(options) % len(remaining_divisors)]
        else:
            b = divisors[len(options) % len(divisors)]

        if a not in used_dividends and f"{a} ÷ {b}" not in correct_answers and a % b == 0 and a // b != quotient:
            options.append(f"{a} ÷ {b}")
            used_dividends.add(a)
            divisor_usage[b] += 1

        # Update remaining_divisors if any divisor is used up
        remaining_divisors = [d for d in divisors if divisor_usage[d] < (num_options + len(divisors) - 1) // len(divisors)]
        attempts += 1

    if len(options) < num_options:
        return None

    return {
        "quotient": quotient,
        "options": options,
        "correct_answers": list(correct_answers)
    }


def generate_div_quotient_with_remainder(min_a, max_a, num_options, divisors):
    # Determine the maximum number of correct options based on the number of divisors
    max_correct_options = min(len(divisors), 4)
    num_correct = random.choices(range(1, max_correct_options + 1), k=1)[0]

    options = []
    correct_answers = set()
    used_dividends = set()
    quotient = None

    divisor_quotient_map = {}

    # Create a map of quotients to potential dividend and divisor pairs
    for divisor in divisors:
        for a in range(min_a, max_a + 1):
            if a % divisor != 0 and a > divisor:
                q = a // divisor
                if q not in divisor_quotient_map:
                    divisor_quotient_map[q] = []
                divisor_quotient_map[q].append((a, divisor))

    while len(correct_answers) < num_correct:
        if not divisor_quotient_map:
            return None
        quotient = random.choice(list(divisor_quotient_map.keys()))
        potential_pairs = divisor_quotient_map[quotient]
        if len(potential_pairs) >= num_correct:
            selected_pairs = random.sample(potential_pairs, num_correct)
            valid_selection = True
            used_divisors = set()
            for a, b in selected_pairs:
                if a in used_dividends or b in used_divisors:
                    valid_selection = False
                    break
                used_divisors.add(b)
            if valid_selection:
                for a, b in selected_pairs:
                    correct_answers.add(f"{a} ÷ {b}")
                    options.append(f"{a} ÷ {b}")
                    used_dividends.add(a)
                break

    # Ensure we have correct answers before generating incorrect options
    if not correct_answers:
        return None

    # Track the usage of each divisor
    used_divisors = [int(pair.split(' ÷ ')[1]) for pair in correct_answers]
    remaining_divisors = [d for d in divisors if d not in used_divisors]

    # Distribute divisors for incorrect options to ensure even distribution
    divisor_usage = {div: 0 for div in divisors}
    for div in used_divisors:
        divisor_usage[div] += 1

    attempts = 0
    while len(options) < num_options and attempts < 10000:
        a = random.randint(min_a, max_a)
        if remaining_divisors:
            b = remaining_divisors[len(options) % len(remaining_divisors)]
        else:
            b = divisors[len(options) % len(divisors)]

        if a > b and a not in used_dividends and f"{a} ÷ {b}" not in correct_answers and a % b != 0 and a // b != quotient:
            options.append(f"{a} ÷ {b}")
            used_dividends.add(a)
            divisor_usage[b] += 1

        # Update remaining_divisors if any divisor is used up
        remaining_divisors = [d for d in divisors if divisor_usage[d] < (num_options + len(divisors) - 1) // len(divisors)]
        attempts += 1

    if len(options) < num_options:
        return None

    return {
        "quotient": quotient,
        "options": options,
        "correct_answers": list(correct_answers)
    }
    
def generate_quotient_rem(min_a, max_a, divisors):
    attempts = 1000
    while attempts > 0:
        a = random.randint(min_a, max_a)
        b = random.choice(divisors)
        
        if a % b != 0 and a > b:
            quotient = a // b
            return {
                'a': a,
                'b': b,
                'quotient': quotient
            }
        
        attempts -= 1
    
     
def generate_remainder_options(min_a, max_a, num_options, divisors):
    # Determine the maximum number of correct options based on the number of divisors
    max_correct_options = min(len(divisors), 4)
    num_correct = random.choices(range(1, max_correct_options + 1), k=1)[0]

    options = set()
    correct_answers = set()
    used_divisors = set()
    used_dividends = set()
    remainder = None

    divisor_remainder_map = {}

    # Create a map of remainders to potential dividend and divisor pairs
    for divisor in divisors:
        for a in range(min_a, max_a + 1):
            r = a % divisor
            if r != 0:
                if r not in divisor_remainder_map:
                    divisor_remainder_map[r] = []
                divisor_remainder_map[r].append((a, divisor))

    while len(correct_answers) < num_correct:
        if not divisor_remainder_map:
            return None
        remainder = random.choice(list(divisor_remainder_map.keys()))
        if len(divisor_remainder_map[remainder]) >= num_correct:
            selected_pairs = random.sample(divisor_remainder_map[remainder], num_correct)
            valid_selection = True
            selected_divisors = {pair[1] for pair in selected_pairs}
            if len(selected_divisors) != num_correct:
                valid_selection = False
            if valid_selection:
                for a, b in selected_pairs:
                    correct_answers.add(f"{a} ÷ {b}")
                    options.add(f"{a} ÷ {b}")
                    used_divisors.add(b)
                    used_dividends.add(a)
                break

    # Ensure we have correct answers before generating incorrect options
    if not correct_answers:
        return None

    # Generate incorrect options
    attempts = 0
    incorrect_remainders = set()
    while len(options) < num_options and attempts < 10000:
        a = random.randint(min_a, max_a)
        available_divisors = [d for d in divisors if d not in used_divisors]
        if not available_divisors:
            available_divisors = divisors  # Reset if all divisors are used

        b = random.choice(available_divisors)
        r = a % b
        if (a not in used_dividends and
                f"{a} ÷ {b}" not in correct_answers and
                r != remainder and
                r > 0 and
                r not in incorrect_remainders):
            options.add(f"{a} ÷ {b}")
            used_divisors.add(b)
            used_dividends.add(a)
            incorrect_remainders.add(r)
        attempts += 1

    if len(options) < num_options:
        return None

    return {
        "remainder": remainder,
        "options": list(options),
        "correct_answers": list(correct_answers)
    }

def is_valid_with_carry(a, b):
    str_a = str(a)
    len_a = len(str_a)
    
    # Ensure that at least one digit of `a` when multiplied by `b` results in a carry over
    carry_found = False
    for i in range(len_a):
        if int(str_a[i]) * b >= 10:
            carry_found = True
        else:
            return False  # If any digit does not result in a carry, it's invalid
    
    return carry_found

def is_valid_without_carry(a, b):
    str_a = str(a)
    len_a = len(str_a)

    # Check that each digit (except the last one) of `a` is less than `b`
    for i in range(len_a - 1):
        if int(str_a[i]) * b >= 10:
            return False

    # Check that the product of the last digit of `a` and `b` is less than 10
    if int(str_a[-1]) * b >= 10:
        return False

    return True

def generate_without_carry_options(min_x, max_x, min_y, max_y, num_options=4, d=50):
    attempts = 0
    options = set()
    used_pairs = set()
    used_b_values = set()

    while attempts < 10000:  # Limit the number of retries to prevent infinite loops
        while len(options) < num_options:
            a = random.randint(min_x, max_x)
            b = random.randint(min_y, max_y)

            if (a, b) in used_pairs or (b, a) in used_pairs or b in used_b_values:
                continue  # Skip if the pair or the b value has been used

            if is_valid_without_carry(a, b):
                # Randomize the format of the option
                if random.choice([True, False]):
                    option = f"{a} × {b}"
                else:
                    option = f"{b} × {a}"
                options.add(option)
                used_pairs.add((a, b))  # Mark the pair as used
                used_pairs.add((b, a))  # Mark the reverse pair as used
                used_b_values.add(b)    # Mark the b value as used

        if len(options) == num_options:
            products = [int(option.split(' × ')[0]) * int(option.split(' × ')[1]) for option in options]
            if max(products) - min(products) <= d:
                # Split options into separate (a, b) pairs for each generated option
                result = {
                    f"a{i+1}": int(option.split(' × ')[0]) for i, option in enumerate(options)
                }
                result.update({
                    f"b{i+1}": int(option.split(' × ')[1]) for i, option in enumerate(options)
                })
                result["options"] = list(options)
                return result
        
        attempts += 1
        options = set()  # Reset options for next attempt
        used_pairs = set()  # Reset used pairs for next attempt
        used_b_values = set()  # Reset used b values for next attempt

    return None

def generate_close_negatives(min_range=-100, max_range=0, max_difference=10):
    """
    Generates four distinct integers within a given range,
    ensuring that the absolute difference between any two integers is no more than a specified value.
    Args:
    min_range (int): The minimum value for the range.
    max_range (int): The maximum value for the range.
    max_difference (int): The maximum allowed difference between any two values.
    Returns:
    list: A list containing the generated integers.
    """
    if max_difference <= 0:
        raise ValueError("max_difference must be greater than 0")
    if min_range > max_range:
        raise ValueError("min_range must be less than or equal to max_range")

    def generate_within_range():
        return random.randint(min_range, max_range)

    def ensure_closeness(numbers):
        unique_numbers = set(numbers)
        while len(unique_numbers) < 4 or any(abs(x - y) > max_difference for x in unique_numbers for y in unique_numbers if x != y):
            unique_numbers = {generate_within_range() for _ in range(4)}
        return list(unique_numbers)

    numbers = ensure_closeness([generate_within_range() for _ in range(4)])
    random.shuffle(numbers)
    return numbers

def generate_mixed_numbers(min_range=-100, max_range=100, max_difference=10):
    """
    Generates four distinct integers within a given range,
    ensuring that the absolute difference between any two integers is no more than a specified value,
    and includes both negative and positive numbers.
    Args:
    min_range (int): The minimum value for the range.
    max_range (int): The maximum value for the range.
    max_difference (int): The maximum allowed difference between any two values.
    Returns:
    list: A list containing the generated integers.
    """
    if max_difference <= 0:
        raise ValueError("max_difference must be greater than 0")
    if min_range > max_range:
        raise ValueError("min_range must be less than or equal to max_range")

    def generate_within_range():
        return random.randint(min_range, max_range)

    def ensure_closeness(numbers):
        unique_numbers = set(numbers)
        while (len(unique_numbers) < 4 or 
               any(abs(x - y) > max_difference for x in unique_numbers for y in unique_numbers if x != y) or 
               (any(x < 0 for x in unique_numbers) == False or any(x > 0 for x in unique_numbers) == False)):
            unique_numbers = {generate_within_range() for _ in range(4)}
        return list(unique_numbers)

    numbers = ensure_closeness([generate_within_range() for _ in range(4)])
    random.shuffle(numbers)
    return numbers

def generate_abs_mixed_numbers(min_range, max_range, max_difference):
    # Generate four unique numbers within the specified range
    while True:
        numbers = [random.randint(min_range, max_range) for _ in range(4)]
        if len(set(numbers)) == 4 and all(abs(x - y) <= max_difference for x in numbers for y in numbers if x != y):
            break

    # Randomly choose at least two positions to have absolute value notation
    abs_count = random.choice([2, 3, 4])  # Choose randomly between 2, 3, or 4 options to have absolute notation
    abs_indices = random.sample(range(4), k=abs_count)

    # Apply absolute value notation to the chosen indices
    abs_numbers = []
    for i in range(4):
        if i in abs_indices:
            number = abs(numbers[i]) if random.choice([True, False]) else -abs(numbers[i])
            abs_numbers.append(f"|{number}|")
        else:
            abs_numbers.append(numbers[i])
            
    random.shuffle(abs_numbers)
    return abs_numbers

def generate_pos_mixed_numbers(min_range, max_range, num_options):
    while True:
        # Generate unique numbers within the specified range
        numbers = [random.randint(min_range, max_range) for _ in range(num_options)]
        
        # Ensure all numbers are unique
        if len(set(numbers)) == num_options:
            break
    
    formatted_numbers = []
    for number in numbers:
        if number > 0:
            formatted_numbers.append(f"+{number}")
        else:
            formatted_numbers.append(str(number))
    
    random.shuffle(formatted_numbers)
    return formatted_numbers

def generate_neg_mixed_numbers(min_range, max_range, num_options):
    while True:
        # Generate unique numbers within the specified range, including 0
        numbers = [random.randint(min_range, max_range) for _ in range(num_options)]
        
        # Ensure all numbers are unique
        if len(set(numbers)) == num_options:
            # Ensure at least one number can be zero
            if 0 in numbers or num_options < (max_range - min_range + 1):
                break

    formatted_numbers = []
    for number in numbers:
        if number < 0:
            formatted_numbers.append(f"{number}")
        elif number == 0:
            formatted_numbers.append("0")
        else:
            formatted_numbers.append(f"+{number}")
    
    random.shuffle(formatted_numbers)
    return formatted_numbers

def generate_mixed_numbers_new(min_range, max_range, num_options):
    while True:
        # Generate unique numbers within the specified range
        numbers = [random.randint(min_range, max_range) for _ in range(num_options)]
        
        # Ensure all numbers are unique
        if len(set(numbers)) == num_options:
            # Check if there's at least one positive and one negative number
            if any(n > 0 for n in numbers) and any(n < 0 for n in numbers):
                break

    formatted_numbers = []
    for number in numbers:
        if number > 0:
            formatted_numbers.append(f"+{number}" if random.choice([True, False]) else str(number))
        elif number < 0:
            formatted_numbers.append(str(number))
        else:
            formatted_numbers.append("0")
    
    random.shuffle(formatted_numbers)
    return formatted_numbers

def generate_mixed_numbers_special(min_range, max_range, num_options):
    while True:
        # Generate unique numbers within the specified range
        numbers = [random.randint(min_range, max_range) for _ in range(num_options)]
        
        # Ensure all numbers are unique
        if len(set(numbers)) == num_options:
            # Check if there's at least one positive and one negative number
            if any(n > 0 for n in numbers) and any(n < 0 for n in numbers):
                break

    # Count positive numbers
    positive_numbers = [num for num in numbers if num > 0]
    
    if len(positive_numbers) > 1 and len(positive_numbers) < 4:
        # Ensure not all positive integers are formatted as -(-number) or -(−number)
        num_pos_formatted = random.randint(1, len(positive_numbers) - 1)
        pos_formatted = random.sample(positive_numbers, num_pos_formatted)
    else:
        pos_formatted = positive_numbers

    formatted_numbers = []
    for number in numbers:
        if number > 0:
            if number in pos_formatted:
                # Format as -(-number) or -(−number) but not all positive numbers
                formatted_numbers.append(f"-(-{number})" if random.choice([True, False]) else f"-({-number})")
            else:
                formatted_numbers.append(str(number))
        elif number < 0:
            formatted_numbers.append(str(number))
        else:
            formatted_numbers.append("0")
    
    random.shuffle(formatted_numbers)
    return formatted_numbers

def generate_p_q_for_case_addition(case):
    p, q = 0, 0  # Default values
    if case == 'a':
        p = random.randint(1, 50)
        q = random.randint(0, 50 - p)  # Ensure p + q <= 50
    elif case == 'b':
        p = random.randint(-50, -1)
        q = random.randint(- 50 - p, 0)  # Ensure p + q <= 50
    elif case == 'c':
        p = random.randint(1, 50)
        q = random.randint(-p+1, 0)
    elif case == 'd':
        p = random.randint(-50, -1)
        q = random.randint(0, -p-1)
    return p, q

def generate_p_q_for_case_subtraction(case):
    p, q = 0, 0  # Default values
    if case == 'a':
        p = random.randint(1, 50)
        q = random.randint(0, 50 - p)
    elif case == 'b':
        p = random.randint(-50, -1)
        q = random.randint(- 50 - p, 0)  # Ensure p + q <= 50
    elif case == 'c':
        p = random.randint(-49, 0)
        q = random.randint(- 50, p-1)  # Ensure p + q <= 50
    elif case == 'd':
        p = random.randint(1, 50)
        q = random.randint(-p+1, 0)
    elif case == 'e':
        p = random.randint(-50, -1)
        q = random.randint(0, -p-1)
    return p, q

def generate_addition_abs_mixed_numbers():
    cases = ['a', 'b', 'c', 'd']
    options = []
    p, q = 0, 0  # Default values

    # Generate unique options for each case
    for case in cases:
        attempts = 0
        while attempts < 10000:  # Attempt up to 50 times
            p, q = generate_p_q_for_case_addition(case)
            abs_value = abs(p + q)
            if abs_value not in [abs(opt[0]) for opt in options]:
                options.append((abs_value, f"|({p}) + ({q})|"))
                break
            attempts += 1

    # Shuffle options to randomize order
    random.shuffle(options)

    # Extract formatted options in the order they were generated
    formatted_options = [opt[1] for opt in options]

    return formatted_options

def generate_subtraction_abs_mixed_numbers():
    cases = ['a', 'b', 'c', 'd', 'e']
    options = []
    p, q = 0, 0  # Default values

    # Generate unique options for each case
    for case in cases:
        attempts = 0
        while attempts < 10000:  # Attempt up to 50 times
            p, q = generate_p_q_for_case_subtraction(case)
            abs_value = abs(p - q)
            if abs_value not in [abs(opt[0]) for opt in options]:
                options.append((abs_value, f"|({p}) - ({q})|"))
                break
            attempts += 1

    # Shuffle options to randomize order
    random.shuffle(options)

    # Extract formatted options in the order they were generated
    formatted_options = [opt[1] for opt in options]

    return formatted_options

def generate_mul_options_single(min_a, max_a, num_options, digits_y, t):
    attempts = 0
    options = set()
    products = set()

    while attempts < 10000:  # Limit the number of retries to prevent infinite loops
        while len(options) < num_options:
            a = random.randint(min_a, max_a)
            b = random.choice(digits_y)  # Choose a random digit for each option
            if 1 <= a <= 9 and 1 <= b <= 9:  # Ensure both a and b are 1-digit numbers
                pair = [a, b]
                random.shuffle(pair)  # Randomize the order of a and b
                a, b = pair
                product = a * b
                if product not in products:
                    option = f"{a} × {b}"
                    if option not in options:
                        options.add(option)
                        products.add(product)

        if len(options) == num_options:
            max_product = max(products)
            min_product = min(products)
            if max_product - min_product <= t:
                # Split options into separate (a, b) pairs for each generated option
                result = {
                    f"a{i+1}": int(option.split(' × ')[0]) for i, option in enumerate(options)
                }
                result.update({
                    f"b{i+1}": int(option.split(' × ')[1]) for i, option in enumerate(options)
                })
                result["options"] = list(options)
                return result

        attempts += 1
        options = set()  # Reset options for next attempt
        products = set()  # Reset products for next attempt

    return None

def generate_addition_option(min_value, max_value):
    while True:
        p = random.randint(min_value, max_value)
        q = random.randint(min_value, max_value)
        
        # Ensure p and q have the same sign and are not equal
        if (p == 0 or q == 0 or (p > 0 and q > 0) or (p < 0 and q < 0)) and p != q:
            s = p + q
            if min_value <= s <= max_value:  # Ensure s is within the range
                equation_type = random.choice(["p", "q"])
                if equation_type == "p":
                    option = f"({p}) + _____ = {s}"
                    c = q  # c is the correct answer for the missing value
                else:
                    option = f"_____ + ({q}) = {s}"
                    c = p  # c is the correct answer for the missing value
                
                result = {
                    "p": f"({p})" if equation_type == "p" else "__",
                    "q": f"({q})" if equation_type == "q" else "__",
                    "s": s,
                    "c": c,
                    "option": option
                }
                return result

def generate_addition_option_with_constraints(min_p=1, max_p=50, min_q=-50, max_q=-1):
    while True:
        p = random.randint(min_p, max_p)
        q = random.randint(min_q, max_q)
        
        # Ensure |p| > |q| and that p is positive and q is negative
        if abs(p) > abs(q) and p > 0 and q < 0:
            s = p + q
            # Ensure s is within the range
            if min_p + min_q <= s <= max_p + max_q:
                equation_type = random.choice(["p", "q"])
                if equation_type == "p":
                    option = f"({p}) + _____ = {s}"
                    c = q  # c is the correct answer for the missing value
                else:
                    option = f"_____ + ({q}) = {s}"
                    c = p  # c is the correct answer for the missing value
                
                result = {
                    "p": f"({p})" if equation_type == "p" else "__",
                    "q": f"({q})" if equation_type == "q" else "__",
                    "s": s,
                    "c": c,
                    "option": option
                }
                return result

def generate_addition_option_with_opposite_constraints(min_p, max_p, min_q, max_q):
    while True:
        p = random.randint(min_p, max_p)
        q = random.randint(min_q, max_q)
        
        # Ensure p is positive, q is negative, and |p| < |q|
        if p > 0 and q < 0 and abs(p) < abs(q):
            s = p + q
            if min_p <= p <= max_p and min_q <= q <= max_q:
                # Generate equation options
                equation_type = random.choice(["p", "q"])
                if equation_type == "p":
                    option = f"({p}) + _____ = {s}"
                    c = q  # c is the correct answer for the missing value
                else:
                    option = f"_____ + ({q}) = {s}"
                    c = p  # c is the correct answer for the missing value
                
                result = {
                    "p": f"({p})" if equation_type == "p" else "__",
                    "q": f"({q})" if equation_type == "q" else "__",
                    "s": s,
                    "c": c,
                    "option": option
                }
                return result

def generate_wrong_multiples(X, num_multiples=9):
    """
    Generates random multiples of X plus a random number from 1 to X-1.

    Args:
    X (int): The base number to generate multiples of.
    num_multiples (int): The number of multiples to generate.

    Returns:
    list: A list of integers, each being a random multiple of X plus a random number from 1 to X-1.
    """
    if X <= 1:
        raise ValueError("X must be greater than 1")

    multiples = set()
    while len(multiples) < num_multiples:
        multiple = X * random.randint(1, 10) + random.randint(1, X-1)
        multiples.add(multiple)

    return list(multiples)

def generate_with_carry_options(min_x, max_x, min_y, max_y, num_options=4, d=50):
    attempts = 0
    options = set()
    used_pairs = set()
    used_b_values = set()

    while attempts < 10000:  # Limit the number of retries to prevent infinite loops
        while len(options) < num_options:
            a = random.randint(min_x, max_x)
            b = random.randint(min_y, max_y)

            if (a, b) in used_pairs or (b, a) in used_pairs or b in used_b_values:
                continue  # Skip if the pair or the b value has been used

            if is_valid_with_carry(a, b):
                # Randomize the format of the option
                if random.choice([True, False]):
                    option = f"{a} × {b}"
                else:
                    option = f"{b} × {a}"
                options.add(option)
                used_pairs.add((a, b))  # Mark the pair as used
                used_pairs.add((b, a))  # Mark the reverse pair as used
                used_b_values.add(b)    # Mark the b value as used

        if len(options) == num_options:
            products = [int(option.split(' × ')[0]) * int(option.split(' × ')[1]) for option in options]
            if max(products) - min(products) <= d:
                # Split options into separate (a, b) pairs for each generated option
                result = {
                    f"a{i+1}": int(option.split(' × ')[0]) for i, option in enumerate(options)
                }
                result.update({
                    f"b{i+1}": int(option.split(' × ')[1]) for i, option in enumerate(options)
                })
                result["options"] = list(options)
                return result
        
        attempts += 1
        options = set()  # Reset options for next attempt
        used_pairs = set()  # Reset used pairs for next attempt
        used_b_values = set()  # Reset used b values for next attempt

    return None

def generate_div_options(min_a, max_a, num_options, d, divisors):
    attempts = 0
    options = set()
    used_dividends = set()
    used_quotients = set()

    while attempts < 10000:  # Limit the number of retries to prevent infinite loops
        options.clear()
        used_dividends.clear()
        used_quotients.clear()

        if len(divisors) >= num_options:
            selected_divisors = random.sample(divisors, num_options)
        else:
            selected_divisors = []
            for i in range(num_options):
                selected_divisors.append(divisors[i % len(divisors)])

        for b in selected_divisors:
            while True:
                a = random.randint(min_a, max_a)
                quotient = a // b
                if b != 0 and a % b == 0 and a > b and (a, b) not in used_dividends and quotient not in used_quotients:
                    option = f"{a} ÷ {b}"
                    options.add(option)
                    used_dividends.add((a, b))
                    used_quotients.add(quotient)
                    break

        if len(options) == num_options:
            quotients = [int(option.split(' ÷ ')[0]) // int(option.split(' ÷ ')[1]) for option in options]
            if max(quotients) - min(quotients) <= d:
                # Split options into separate (a, b) pairs for each generated option
                result = {
                    f"a{i+1}": int(option.split(' ÷ ')[0]) for i, option in enumerate(options)
                }
                result.update({
                    f"b{i+1}": int(option.split(' ÷ ')[1]) for i, option in enumerate(options)
                })
                result["options"] = list(options)
                return result

        attempts += 1

    return None

def generate_div_options_remainder(min_a, max_a, num_options, d, divisors):
    attempts = 0
    options = set()
    used_dividends = set()
    used_quotients = set()

    while attempts < 10000:  # Limit the number of retries to prevent infinite loops
        options.clear()
        used_dividends.clear()
        used_quotients.clear()

        if len(divisors) >= num_options:
            selected_divisors = random.sample(divisors, num_options)
        else:
            selected_divisors = []
            for i in range(num_options):
                selected_divisors.append(divisors[i % len(divisors)])

        for b in selected_divisors:
            while True:
                a = random.randint(min_a, max_a)
                quotient = a // b
                if b != 0 and a % b != 0 and a > b and (a, b) not in used_dividends and quotient not in used_quotients:
                    option = f"{a} ÷ {b}"
                    options.add(option)
                    used_dividends.add((a, b))
                    used_quotients.add(quotient)
                    break

        if len(options) == num_options:
            quotients = [int(option.split(' ÷ ')[0]) // int(option.split(' ÷ ')[1]) for option in options]
            if max(quotients) - min(quotients) <= d:
                # Split options into separate (a, b) pairs for each generated option
                result = {
                    f"a{i+1}": int(option.split(' ÷ ')[0]) for i, option in enumerate(options)
                }
                result.update({
                    f"b{i+1}": int(option.split(' ÷ ')[1]) for i, option in enumerate(options)
                })
                result["options"] = list(options)
                return result
        
        attempts += 1

    return None

def set_clock_delta(x_list, y_list):
    """Converts T1 into minutes, generates T2, and calculates the time delta."""
    # Randomly select values from x_list and y_list
    context = {}
    x = random.choice(x_list)
    y = random.choice(y_list)
    
    # Convert selected hours and minutes to total minutes
    t1_minutes = 60 * x + y
    
    # Generate a random current time T2 in the format P:Q where P is hours and Q is minutes
    p = random.randint(1, 12)
    q = random.randint(0, 59)
    t2_minutes = 60 * p + q

    x = str(x)
    y = str(y)
    p = str(p)
    q = str(q)
    
    # Randomly decide whether to add or subtract T1 from T2
    time_delta_type = random.choice(['before', 'after'])
    if time_delta_type == 'before':
        result_time = t2_minutes - t1_minutes
    else:
        result_time = t2_minutes + t1_minutes

    if result_time >= 13*60:
        result_time -= 12*60

    if result_time < 0:
        result_time += 12*60

    result_hours = (result_time // 60) % 12
    result_minutes = result_time % 60

    if x == "0":
        x = "12"
    if len(y) == 1:
        y = "0" + y

    if p == "0":
        p = "12"
    if len(q) == 1:
        q = "0" + q

    # Ensure the result time is within the 12-hour clock format
    result_hours = 12 if result_hours == 0 else result_hours
    context.update({
        'time_delta': time_delta_type,
        'T2': t2_minutes,
        'T1': t1_minutes,
        'result_time': result_time,
        'h1': x,
        'm1': y,
        'h2': p,
        'm2': q,
        'result_hrs': result_hours,
        'result_min': result_minutes,
    })
    return context

def find_multiples_in_range(start, end, step, exclude):
    multiples = []
    for i in range(start, end + 1, step):
        if exclude != 0 and i % exclude == 0:
            continue
        multiples.append(i)
    return multiples

def generate_perc_values(y_start, y_end, y_step, a_start, a_end, a_step, y_exclude=0, a_exclude=0):
    """
    Generate x and z values such that 100 * z / x is a natural number.
    a = 100 * z / x should be an integer within the given range.
    """
    y_values = find_multiples_in_range(y_start, y_end, y_step, y_exclude)
    a_values = find_multiples_in_range(a_start, a_end, a_step, a_exclude)

    while True:
        y = random.choice(y_values)
        a = random.choice(a_values)
        
        z = (a * y) // 100  # Ensure z is an integer
        
        if (100 * z) % y == 0 and z > 0:  # Ensure z is positive
            return y, z, a
 
def generate_perc_factor_values(y_start, y_end, y_step, a_start, a_end, a_step, y_exclude=0, a_exclude=0):
    """
    Generate y and z values such that 100 * z / y is a natural number.
    y is a factor of z and 100z/y is an integer within the given range.
    """
    y_values = find_multiples_in_range(y_start, y_end, y_step, y_exclude)
    a_values = find_multiples_in_range(a_start, a_end, a_step, a_exclude)

    while True:
        y = random.choice(y_values)
        a = random.choice(a_values)
        
        z = (a * y) // 100  # Ensure z is an integer

        if (100 * z) % y == 0 and z > 0:  # Ensure z is positive
            return y, z, a  
             
def generate_xy_values(x_start, x_end, x_step, y_start, y_end, y_step, x_exclude=0, y_exclude=0):
    """
    Generate x and y values such that x * y is divisible by 100.
    """
    x_values = find_multiples_in_range(x_start, x_end, x_step, x_exclude)
    y_values = find_multiples_in_range(y_start, y_end, y_step, y_exclude)

    if not x_values or not y_values:
        raise ValueError("No valid values for x or y within the given range and constraints")

    valid_pairs = [(x, y) for x in x_values for y in y_values if (x * y) % 100 == 0]

    if not valid_pairs:
        raise ValueError("No valid x, y pairs found within the given constraints")

    return random.choice(valid_pairs)

def generate_perc_alternate_mul(min_a, max_a, min_x, max_x):
    while True:
        a = random.randint(min_a, max_a)
        x = random.randint(min_x, max_x)
        
        # Ensure one of A or X is a multiple of 100, but not both
        if (a % 100 == 0 and x % 100 != 0) or (a % 100 != 0 and x % 100 == 0):
            # Ensure AX is divisible by 100
            if (a * x) % 100 == 0:
                return {
                    "a": a,
                    "x": x,
                }

def generate_ratios_a_constant(a_range_start, a_range_end, b_range_start, b_range_end):
    """
    Generate 4 distinct ratio options with 'a' constant and 'b' in the given range.
    
    Parameters:
    a_range_start (int): Start of the range for 'a'.
    a_range_end (int): End of the range for 'a'.
    b_range_start (int): Start of the range for 'b'.
    b_range_end (int): End of the range for 'b'.
    
    Returns:
    dict: A dictionary with generated ratios and the components used.
    """
    options = set()
    a = random.randint(a_range_start, a_range_end)
    b_values = set()

    while len(b_values) < 4:
        b_value = random.randint(b_range_start, b_range_end)
        if b_value not in b_values:
            b_values.add(b_value)
            options.add(f"{a}:{b_value}")
    
    options = list(options)
    result = {f"a{i+1}": int(option.split(':')[0]) for i, option in enumerate(options)}
    result.update({f"b{i+1}": int(option.split(':')[1]) for i, option in enumerate(options)})
    result["options"] = options
    
    return result

def generate_ratios_b_constant(a_range_start, a_range_end, b_range_start, b_range_end):
    """
    Generate 4 distinct ratio options with 'b' constant and 'a' in the range [1, 2b].
    
    Parameters:
    a_range_start (int): Start of the range for 'a'.
    a_range_end (int): End of the range for 'a'.
    b_range_start (int): Start of the range for 'b'.
    b_range_end (int): End of the range for 'b'.
    
    Returns:
    dict: A dictionary with generated ratios and the components used.
    """
    options = set()
    b = random.randint(b_range_start, b_range_end)
    a_values = set()

    while len(a_values) < 4:
        a_value = random.randint(1, 2 * b)
        if a_value not in a_values:
            a_values.add(a_value)
            options.add(f"{a_value}:{b}")
    
    options = list(options)
    result = {f"a{i+1}": int(option.split(':')[0]) for i, option in enumerate(options)}
    result.update({f"b{i+1}": int(option.split(':')[1]) for i, option in enumerate(options)})
    result["options"] = options
    
    return result

def generate_ratios_divisors():
    """
    Generate 4 distinct ratio options based on predefined b values,
    ensuring unique a/b values (up to two decimal places) and unique a values.
    
    Returns:
    dict: A dictionary with generated ratios and the components used.
    """
    b_list = [2, 3, 4, 5, 6, 8, 9, 10]
    selected_bs = random.sample(b_list, 4)  # Ensure 4 unique b values are chosen
    options = set()
    ratios = set()  # To keep track of unique a/b values up to two decimal places
    used_a_values = set()  # To keep track of unique a values

    for b in selected_bs:
        while True:
            a = random.randint(1, 2 * b)
            ratio = round(a / b, 2)  # Round the ratio to two decimal places
            if ratio not in ratios and a not in used_a_values:
                ratios.add(ratio)
                used_a_values.add(a)
                options.add(f"{a}:{b}")
                break  # Break the loop when a unique ratio is found

    result = {f"a{i+1}": int(option.split(':')[0]) for i, option in enumerate(options)}
    result.update({f"b{i+1}": int(option.split(':')[1]) for i, option in enumerate(options)})
    result["options"] = list(options)
    
    return result

def generate_ratios_with_denoms(min_range, max_range):
    """
    Generate 4 distinct ratio options with unique 'a' and 'b' values based on specified range.
    
    Parameters:
    min_range (int): Minimum range for 'a' and 'b'.
    max_range (int): Maximum range for 'a' and 'b'.
    
    Returns:
    dict: A dictionary with generated ratios and the components used.
    """
    options = set()
    unique_ratios = set()
    unique_denoms = set()
    used_a_values = set()

    while len(options) < 4:
        a = random.randint(min_range, max_range)
        b_values = [val for val in range(min_range, max_range + 1) if val > a]
        
        if not b_values:
            continue  # If no valid 'b' values, regenerate 'a'

        b = random.choice(b_values)

        denoms = [a, b, 2 * a, 2 * b]

        for denom in denoms:
            a_value = random.randint(1, 2 * denom)
            ratio = a_value / denom

            if ratio not in unique_ratios and denom not in unique_denoms and a_value not in used_a_values:
                unique_ratios.add(ratio)
                unique_denoms.add(denom)
                used_a_values.add(a_value)
                options.add(f"{a_value}:{denom}")
                if len(options) >= 4:
                    break
    
    options = list(options)
    result = {f"a{i+1}": int(option.split(':')[0]) for i, option in enumerate(options)}
    result.update({f"b{i+1}": int(option.split(':')[1]) for i, option in enumerate(options)})
    result["options"] = options

    return result

def generate_ratios_equivalent():
    """
    Generate 2 ratios a:b and c:d (b,d≠0) which are equivalent 50% of the time and not equivalent 50% of the time.
    Ratios are equivalent IFF ad=bc. Let’s have a,b,c,d<20.
    
    Returns:
    dict: A dictionary with generated ratios and the components used.
    """
    def get_unique_values():
        values = list(range(2, 20))  # start from 2 to ensure unique values
        random.shuffle(values)
        selected_values = values[:3]  # select three values
        selected_values.append(1)  # add 1 to the list
        random.shuffle(selected_values)  # shuffle to ensure randomness
        return selected_values

    def is_valid_ratio(a, b, c, d):
        ratio_1 = a / b
        ratio_2 = c / d
        return (ratio_1 > 1 and ratio_2 > 1) or (ratio_1 < 1 and ratio_2 < 1)

    is_equivalent = random.choice([True, False])
    
    while True:
        values = get_unique_values()
        a, b, c, d = values

        if is_equivalent:
            if a * d == b * c and is_valid_ratio(a, b, c, d):
                break
        else:
            if a * d != b * c and is_valid_ratio(a, b, c, d):
                break

    ratio_1 = f"{a}:{b}"
    ratio_2 = f"{c}:{d}"
    
    return {"ratio_1": ratio_1, "ratio_2": ratio_2, "is_equivalent": is_equivalent}
def generate_abs_value_question():
    """
    Generate a question to determine if the absolute value equation is true or false.
    X is an integer including zero, and -100 ≤ X ≤ 100.
    
    Returns:
    dict: A dictionary with the generated equation, whether it is True or False, and the values of X and Y.
    """
    def get_unique_x():
        return random.randint(-100, 100)

    x = get_unique_x()
    is_true = random.choice([True, False])
    
    if is_true:
        y = abs(x)
        equation = f"|{x}| = {y}"
    else:
        if x > 0:
            y = -abs(x)
        elif x < 0:
            y = -abs(x)
        else:
            y = random.choice([-1, 1])
        equation = f"|{x}| = {y}"
    
    result = {
        "equation": equation,
        "is_true": is_true,
        "correct": [is_true],  # Ensure the correct key is set correctly
        "x": x,
        "y": y
    }
    
    return result

def generate_integer_subtraction_question():
    # Define p and q based on the given cases
    case = random.choice(['a', 'b', 'c', 'd', 'e', 'f'])
    
    if case == 'a':
        # Case (a): Both integers are positive
        p = random.randint(0, 50)
        q = random.randint(0, 50)
        
    elif case == 'b':
        # Case (b): Both integers are negative
        p = -random.randint(0, 50)
        q = -random.randint(0, 50)
        
    elif case == 'c':
        # Case (c): p - positive, q - negative (|p| > |q|)
        p = random.randint(1, 50)
        q = -random.randint(0, p)
        
    elif case == 'd':
        # Case (d): p - negative, q - positive (|q| > |p|)
        p = -random.randint(0, 50)
        q = random.randint(abs(p) + 1, 50)
        
    elif case == 'e':
        # Case (e): p - positive, q - negative (|q| > |p|)
        p = random.randint(0, 50)
        q = -random.randint(abs(p) + 1, 50)
        
    elif case == 'f':
        # Case (f): p - negative, q - positive (|p| > |q|)
        p = -random.randint(1, 50)
        q = random.randint(0, abs(p))
        
    # Calculate the correct result
    correct_result = abs(p - q)
    
    if random.choice([True, False]):
        # True case
        y = correct_result
        is_true = True
    else:
        # False case - select one of the misconceptions
        false_cases = [
            -correct_result,                    # -|p - q|
            0 if correct_result != 0 else None, # 0 for |p - q| ≠ 0
            -1 if correct_result == 0 else None,# -1 for |p - q| = 0
            1 if correct_result == 0 else None, # 1 for |p - q| = 0
            abs(p) - abs(q) if abs(p) > abs(q) and abs(p) - abs(q) != correct_result else None, # |p| - |q| ≠ |p - q|
            abs(q) - abs(p) if abs(q) > abs(p) and abs(q) - abs(p) != correct_result else None, # |q| - |p| ≠ |p - q|
            abs(p) + abs(q) if abs(p) + abs(q) != correct_result else None,  # |p| + |q| ≠ |p - q|
            -(abs(p) + abs(q)) if abs(p) + abs(q) != correct_result else None # -(|p| + |q|) ≠ |p - q|
        ]
        false_cases = [case for case in false_cases if case is not None]
        y = random.choice(false_cases)
        is_true = False

    equation = f"|({p}) - ({q})| = {y}"
    
    result = {
        "equation": equation,
        "is_true": is_true,
        "correct": [is_true],  # Ensure the correct key is set correctly
        "p": p,
        "q": q,
        "y": y
    }
    
    return result

def generate_integer_addition_question():
    p = random.randint(-50, 50)
    q = random.randint(-50, 50)
    
    if random.choice([True, False]):
        # True case
        y = abs(p + q)
        is_true = True
    else:
        # False case - select one of the misconceptions
        false_cases = [
            -abs(p + q),                    # -|p + q|
            0 if abs(p + q) != 0 else None, # 0 for |p + q| ≠ 0
            -1 if abs(p + q) == 0 else None,# -1 for |p + q| = 0
            1 if abs(p + q) == 0 else None, # 1 for |p + q| = 0
            abs(p) + abs(q) if abs(p + q) != abs(p) + abs(q) else None,  # |p| + |q| ≠ |p + q|
            -(abs(p) + abs(q)) if abs(p + q) != abs(p) + abs(q) else None, # -(|p| + |q|) ≠ |p + q|
            abs(p) - abs(q) if abs(p) > abs(q) and abs(p) - abs(q) != abs(p + q) else None, # |p| - |q|, |p| > |q| and ≠ |p + q|
            abs(q) - abs(p) if abs(q) > abs(p) and abs(q) - abs(p) != abs(p + q) else None  # |q| - |p|, |q| > |p| and ≠ |p + q|
        ]
        false_cases = [case for case in false_cases if case is not None]
        y = random.choice(false_cases)
        is_true = False

    equation = f"|({p}) + ({q})| = {y}"
    
    result = {
        "equation": equation,
        "is_true": is_true,
        "correct": [is_true],  # Ensure the correct key is set correctly
        "p": p,
        "q": q,
        "y": y
    }
    
    return result

def generate_ratios_reduced():
    def get_unique_values():
        values = list(range(2, 21))  # Ensure values are not 1 and less than 20
        random.shuffle(values)
        return values

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_valid_ratio(a, b, c, d):
        ratio_1 = a / b
        ratio_2 = c / d
        return (ratio_1 > 1 and ratio_2 > 1) or (ratio_1 < 1 and ratio_2 < 1)

    def is_reduced(a, b):
        return gcd(a, b) == 1

    is_equivalent = random.choice([True, False])

    while True:
        e, f = get_unique_values()[:2]
        gcd_ef = gcd(e, f)
        e //= gcd_ef
        f //= gcd_ef

        if e == 1 or f == 1:
            continue

        if is_equivalent:
            scale = random.randint(2, 5)
            a, b = e, f
            c, d = e * scale, f * scale

            if any(x >= 20 for x in [a, b, c, d]) or len(set([a, b, c, d])) < 4:
                continue
        else:
            a, b = e, f  # Ensure one ratio is reduced
            scale = random.randint(2, 5)
            c, d = e * scale, f * scale

            if random.choice([True, False]):
                d += gcd(c, d)
            else:
                b += gcd(a, b)

            if any(x >= 20 for x in [a, b, c, d]) or len(set([a, b, c, d])) < 4:
                continue

        if not is_valid_ratio(a, b, c, d):
            continue

        # Ensure only one of the ratios is in the least reduced form
        if is_reduced(a, b) and is_reduced(c, d):
            continue

        if not is_reduced(a, b) and not is_reduced(c, d):
            continue

        break

    ratio_1 = f"{a}:{b}"
    ratio_2 = f"{c}:{d}"

    return {"ratio_1": ratio_1, "ratio_2": ratio_2, "is_equivalent": is_equivalent}
def generate_ratios_not_reduced(min_val, max_val):
    def get_unique_values():
        values = list(range(min_val, max_val + 1))
        random.shuffle(values)
        return values[:4]

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_valid_ratio(a, b, c, d):
        ratio_1 = a / b
        ratio_2 = c / d
        return (ratio_1 > 1 and ratio_2 > 1) or (ratio_1 < 1 and ratio_2 < 1)

    is_equivalent = random.choice([True, False])

    while True:
        values = get_unique_values()
        e, f = values[:2]

        if e == 1 or f == 1:
            continue

        max_scale1 = max_val // max(e, f)
        max_scale2 = max_val // max(e, f)
        if max_scale1 < 2:
            max_scale1 = 2
        if max_scale2 < 2:
            max_scale2 = 2

        scale1 = random.randint(2, max_scale1)
        scale2 = random.randint(2, max_scale2)

        a, b = e * scale1, f * scale1
        c, d = e * scale2, f * scale2

        # Ensure that neither ratio is already in reduced form
        if gcd(a, b) == 1 or gcd(c, d) == 1:
            continue  # If either ratio is already reduced, regenerate

        if not is_equivalent:
            if random.choice([True, False]):
                d += gcd(b, d)
            else:
                b += gcd(b, d)

            # Ensure the modified ratio is still not in reduced form
            if gcd(a, b) == 1 or gcd(c, d) == 1:
                continue

        if all(x <= max_val for x in [a, b, c, d]) and len(set([a, b, c, d])) == 4 and is_valid_ratio(a, b, c, d):
            break

    ratio_1 = f"{a}:{b}"
    ratio_2 = f"{c}:{d}"

    return {"ratio_1": ratio_1, "ratio_2": ratio_2, "is_equivalent": is_equivalent}
def generate_wrong_options(correct_answer, a, b, x_or_y, find_missing):
    wrong_options = set()
    potential_wrong_options_high = []
    potential_wrong_options_low = []

    # Generate higher priority wrong options
    if find_missing == 'c':
        if (b * x_or_y) % a == 0:
            opt1 = b * x_or_y // a  # AX/B
            if opt1 != correct_answer and 0 < opt1 < 10000:
                potential_wrong_options_high.append(opt1)
        if (a * b) % x_or_y == 0:
            opt2 = a * b // x_or_y  # AB/X
            if opt2 != correct_answer and 0 < opt2 < 10000:
                potential_wrong_options_high.append(opt2)
        opt3 = a * b * x_or_y  # ABX
        if opt3 != correct_answer and opt3 < 10000:
            potential_wrong_options_high.append(opt3)
    else:
        if (a * x_or_y) % b == 0:
            opt1 = a * x_or_y // b  # BY/A
            if opt1 != correct_answer and 0 < opt1 < 10000:
                potential_wrong_options_high.append(opt1)
        if (a * b) % x_or_y == 0:
            opt2 = a * b // x_or_y  # AB/Y
            if opt2 != correct_answer and 0 < opt2 < 10000:
                potential_wrong_options_high.append(opt2)
        opt3 = a * b * x_or_y  # ABY
        if opt3 != correct_answer and opt3 < 10000:
            potential_wrong_options_high.append(opt3)

    # Generate lower priority wrong options
    k = random.randint(1, 6)
    opt4 = correct_answer + k  # P + k
    if opt4 != correct_answer:
        potential_wrong_options_low.append(opt4)
    opt5 = correct_answer * 10  # 10P
    if opt5 != correct_answer and opt5 < 10000:
        potential_wrong_options_low.append(opt5)
    if correct_answer % 2 == 0:
        opt6 = correct_answer // 2  # P/2
        if opt6 != correct_answer and 0 < opt6 < 10000:
            potential_wrong_options_low.append(opt6)

    # Prioritize and select up to 3 wrong options
    for option in potential_wrong_options_high:
        if len(wrong_options) >= 3:
            break
        wrong_options.add(option)

    if len(wrong_options) < 3:
        for option in potential_wrong_options_low:
            if len(wrong_options) >= 3:
                break
            wrong_options.add(option)

    return list(wrong_options)

def generate_ratios_find_value():
    """
    Generate 2 equivalent ratios a:b and c:d (b,d≠0) with one of the values being 1, all values < 20.
    Return the ratios with one value missing (represented by ?) and the correct answer.
    """
    def get_unique_values():
        values = list(range(2, 20))  # start from 2 to ensure unique values
        random.shuffle(values)
        selected_values = values[:3]  # select three values
        selected_values.append(1)  # add 1 to the list
        random.shuffle(selected_values)  # shuffle to ensure randomness
        return selected_values

    def find_correct_answer(a, b, c, d, find_missing):
        if find_missing == 'c':
            # We need to find c in a:b = c:d
            correct_answer = a * d // b
        else:
            # We need to find d in a:b = c:d
            correct_answer = b * c // a
        return correct_answer
    
    while True:
        # Generate four unique values with one of them being 1
        values = get_unique_values()
        a, b, c, d = values

        # Ensure a:b = c:d by scaling a and b within the range
        if a == 1:
            c = random.randint(2, 19)
            d = b * c
        elif b == 1:
            d = random.randint(2, 19)
            c = a * d
        elif c == 1:
            a = random.randint(2, 19)
            b = d * a
        elif d == 1:
            b = random.randint(2, 19)
            a = c * b

        # Ensure values remain within the range and are unique
        if a >= 20 or b >= 20 or c >= 20 or d >= 20 or len(set([a, b, c, d])) < 4:
            continue

        # Generate the ratio with one missing value
        find_missing = random.choice(['c', 'd'])
        correct_answer = find_correct_answer(a, b, c, d, find_missing)

        # Ensure the correct answer is a natural number and not equal to 1
        if correct_answer < 2 or correct_answer >= 20:
            continue

        if find_missing == 'c':
            ratio_2 = f"?:{d}"
        else:
            ratio_2 = f"{c}:?"

        # Generate wrong options
        wrong_options = generate_wrong_options(correct_answer, a, b, c if find_missing == 'd' else d, find_missing)
        if len(wrong_options) < 3:
            continue
        break

    result = {
        "ratio_1": f"{a}:{b}",
        "ratio_2": ratio_2,
        "correct_answer": correct_answer,
        "options": wrong_options
    }

    return result

def generate_ratios_find_reduced():
    """
    Generate 2 equivalent ratios a:b and c:d (b,d≠0) where none of the values are 1,
    all values are < 20, and one ratio is in its lowest reduced form while the other is a multiple of it.
    
    Returns:
    dict: A dictionary with generated ratios and the components used.
    """
    def get_unique_values():
        values = list(range(2, 21))  # ensure values are not 1 and less than 20
        random.shuffle(values)
        return values[:2]

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def find_correct_answer(a, b, c, d, find_missing):
        if find_missing == 'c':
            # We need to find c in a:b = c:d
            correct_answer = a * d // b
        else:
            # We need to find d in a:b = c:d
            correct_answer = b * c // a
        return correct_answer

    while True:
        # Generate two values that are not 1
        e, f = get_unique_values()
        gcd_ef = gcd(e, f)
        e //= gcd_ef
        f //= gcd_ef

        # Ensure the reduced ratio is within bounds
        if any(x >= 20 or x <= 1 for x in [e, f]):
            continue

        scale1 = random.randint(2, 5)
        scale2 = random.randint(2, 5)
        a, b = e * scale1, f * scale1
        c, d = e * scale2, f * scale2

        # Ensure values remain within the range and are unique
        if any(x >= 20 or x <= 1 for x in [a, b, c, d]) or (a == c and b == d):
            continue

        # Generate the ratio with one missing value
        find_missing = random.choice(['c', 'd'])
        correct_answer = find_correct_answer(a, b, c, d, find_missing)

        # Ensure the correct answer is a natural number and not equal to 1
        if correct_answer < 2 or correct_answer >= 20:
            continue

        if find_missing == 'c':
            ratio_2 = f"?:{d}"
        else:
            ratio_2 = f"{c}:?"

        # Generate wrong options
        wrong_options = generate_wrong_options(correct_answer, a, b, c if find_missing == 'd' else d, find_missing)
        if len(wrong_options) < 3:
            continue
        break

    return {
        "ratio_1": f"{e}:{f}",
        "ratio_2": ratio_2,
        "correct_answer": correct_answer,
        "options": wrong_options
    }

def generate_ratios_find_not_reduced(min_val, max_val, max_attempts=1000):
    def get_unique_values():
        values = list(range(min_val, max_val + 1))
        random.shuffle(values)
        return values[:4]

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def find_correct_answer(a, b, c, d, find_missing):
        if find_missing == 'c':
            correct_answer = a * d // b
        else:
            correct_answer = b * c // a
        return correct_answer

    def select_e_f():
        while True:
            e, f = random.randint(2, 9), random.randint(2, 9)
            if e != f:
                return e, f

    attempts = 0
    while attempts < max_attempts:
        e, f = select_e_f()

        available_scales = list(range(2, max_val // max(e, f) + 1))
        if len(available_scales) < 4:
            continue
        random.shuffle(available_scales)
        scale1, scale2 = available_scales[:2]
        scale3, scale4 = available_scales[2:4]

        a, b = e * scale1, f * scale1
        c, d = e * scale3, f * scale4

        if any(value > max_val or value < min_val for value in [a, b, c, d]):
            attempts += 1
            continue

        if len(set([a, b, c, d])) < 4:
            attempts += 1
            continue

        find_missing = random.choice(['c', 'd'])
        correct_answer = find_correct_answer(a, b, c, d, find_missing)

        if correct_answer < min_val or correct_answer > max_val:
            attempts += 1
            continue

        if find_missing == 'c':
            ratio_2 = f"?:{d}"
        else:
            ratio_2 = f"{c}:?"

        # Generate wrong options
        wrong_options = generate_wrong_options(correct_answer, a, b, c if find_missing == 'd' else d, find_missing)
        if len(wrong_options) < 3:
            attempts += 1
            continue

        return {
            "ratio_1": f"{a}:{b}",
            "ratio_2": ratio_2,
            "correct_answer": correct_answer,
            "options": wrong_options,
        }
        
# def generate_equivalent_fractions(min_p, max_p, min_q, max_q, min_multiplier, max_multiplier, multiplier_type, num_options=4):
#     attempts = 0
#     while attempts < 10000:  # Limit the number of retries to prevent infinite loops
#         P = random.randint(min_p, max_p)
#         Q = random.randint(min_q, max_q)
#         if multiplier_type == "integer":
#             multiplier = random.randint(min_multiplier, max_multiplier)
#         elif multiplier_type == "fraction":
#             multiplier = random.randint(min_multiplier, max_multiplier)
#         probabilities = [0.3, 0.4, 0.3, 0.0]  # Distribution for 1, 2, 3, 4 correct options
#         num_correct = random.choices([1, 2, 3, 4], weights=probabilities)[0]
#         options = []
#         correct_answers = []
#         for i in range(num_options):
#             current_is_correct = i < num_correct
#             current_multiplier = multiplier if current_is_correct else random.randint(min_multiplier, max_multiplier)
#             current_P = P if current_is_correct else P * current_multiplier
#             current_Q = Q if current_is_correct else Q * current_multiplier
#             fraction = Fraction(current_P, current_Q)
#             if current_is_correct:
#                 correct_answers.append(fraction)
#             options.append(fraction)
#         while len(set(options)) < num_options:
#             incorrect_fraction = Fraction(random.randint(min_p, max_p), random.randint(min_q, max_q))
#             correct_fraction = correct_answers[0]
#             if incorrect_fraction != correct_fraction:
#                 options.append(incorrect_fraction)
#         if len(options) == num_options:
#             return {
#                 "numerator": P,
#                 "denominator": Q,
#                 "correct_answers": correct_answers,
#                 "options": options
#             }
#         attempts += 1
#     return None  # If no valid configuration is found after 10000 attempts


def generate_equivalent_fractions(min_p, max_p, min_q, max_q, min_multiplier, max_multiplier, multiplier_type, num_options=4):
    return {
        "numerator": 1,
        "denominator": 2,
        "correct_answers": [Fraction(1, 2)],
        "options": [Fraction(1, 2), Fraction(2, 4), Fraction(3, 6), Fraction(4, 9)]
    }
    attempts = 0
    while attempts < 100:  # Reduce the number of retries to prevent excessive loops
        P = random.randint(min_p, max_p)
        Q = random.randint(min_q, max_q)
        
        # Ensure Q is not zero to avoid division by zero in fractions
        if Q == 0:
            continue

        # Handle the multiplier type: integer or fractional
        if multiplier_type == "integer":
            multiplier = random.randint(min_multiplier, max_multiplier)
        elif multiplier_type == "fraction":
            num = random.randint(min_multiplier, max_multiplier)
            den = random.randint(min_multiplier, max_multiplier) or 1  # Avoid zero denominator
            multiplier = Fraction(num, den)
        else:
            continue  # Skip if multiplier type is not recognized

        probabilities = [0.3, 0.4, 0.3, 0.0]  # Adjusted: [1, 2, 3, 4 correct options]
        num_correct = random.choices([1, 2, 3, 4], weights=probabilities)[0]
        options = set()
        correct_answers = []

        # Generate correct options
        while len(correct_answers) < num_correct:
            new_fraction = Fraction(P * multiplier, Q * multiplier)
            if new_fraction not in options:
                options.add(new_fraction)
                correct_answers.append(new_fraction)

        # Generate incorrect options
        while len(options) < num_options:
            incorrect_multiplier = random.randint(min_multiplier, max_multiplier) if multiplier_type == "integer" else Fraction(random.randint(min_multiplier, max_multiplier), random.randint(min_multiplier, max_multiplier) or 1)
            incorrect_fraction = Fraction(P * incorrect_multiplier, Q * incorrect_multiplier)
            if incorrect_fraction not in options:
                options.add(incorrect_fraction)

        if len(options) == num_options:
            return {
                "numerator": P,
                "denominator": Q,
                "correct_answers": list(correct_answers),
                "options": list(options)
            }
        attempts += 1
    return None  # If no valid configuration is found after retries



def parse_assignment(line, context):
    var_names, expression = line.split('=', 1)
    expression = expression.strip()

    # Handle new custom functions and operations
    if 'no_carry_sum' in expression:
        min_a, max_a, min_b, max_b = [evaluate_expression(arg.strip(), context) for arg in expression[len('no_carry_sum('):-1].split(',')]
        values = no_carry_sum_range(min_a, max_a, min_b, max_b)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, values):
            context[var_name] = val
        return
    # Incorporating into the existing code
    elif 'generate_mul_options2' in expression:
        args = [arg.strip() for arg in expression[len('generate_mul_options2('):-1].split(',')]
        min_x = evaluate_expression(args[0].strip(), context)
        max_x = evaluate_expression(args[1].strip(), context)
        digits_y = [evaluate_expression(arg.strip(), context) for arg in args[2].strip().strip('[]').split(',') if arg.strip()]
        num_options = evaluate_expression(args[3].strip(), context)
        unique = evaluate_expression(args[4].strip(), context) if len(args) > 4 else False
        
        values = generate_mul_options2(min_x, max_x, digits_y, num_options, unique)
        
        if values:
            var_pairs = [var.strip() for var in var_names.strip().strip("()").split(',')]
            var_a, var_b = var_pairs
            
            # Assuming the first option is the correct answer to extract variables a and b
            a, b = map(int, values['options'][0].split(' x '))
            
            context[var_a] = a
            context[var_b] = b
        
        return
    
    elif 'no_borrow_diff' in expression:
        min_p, max_p, min_q, max_q = [evaluate_expression(arg.strip(), context) for arg in expression[len('no_borrow_diff('):-1].split(',')]
        values = no_borrow_diff_range(min_p, max_p, min_q, max_q)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, values):
            context[var_name] = val
        return
    elif 'must_carry_sum' in expression:
        min_a, max_a, min_b, max_b = [evaluate_expression(arg.strip(), context) for arg in expression[len('must_carry_sum('):-1].split(',')]
        values = must_carry_sum_range(min_a, max_a, min_b, max_b)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, values):
            context[var_name] = val
        return
    elif 'must_borrow_diff' in expression:
        min_p, max_p, min_q, max_q = [evaluate_expression(arg.strip(), context) for arg in expression[len('must_borrow_diff('):-1].split(',')]
        values = must_borrow_diff_range(min_p, max_p, min_q, max_q)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, values):
            context[var_name] = val
        return
    elif 'no_carry_over_mul' in expression:
        min_x, max_x, min_y, max_y = [evaluate_expression(arg.strip(), context) for arg in expression[len('no_carry_over_mul('):-1].split(',')]
        values = no_carry_over_mul_range(min_x, max_x, min_y, max_y)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, values):
            context[var_name] = val
        return
    elif 'with_carry_over_mul' in expression:
        min_x, max_x, min_y, max_y = [evaluate_expression(arg.strip(), context) for arg in expression[len('with_carry_over_mul('):-1].split(',')]
        values = with_carry_over_mul_range(min_x, max_x, min_y, max_y)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, values):
            context[var_name] = val
        return
    elif 'add_without_carry_over' in expression:
        p, q = [evaluate_expression(arg.strip(), context) for arg in expression[len('add_without_carry_over('):-1].split(',')]
        value = add_without_carry_over(p, q)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name in var_names:
            context[var_name] = value
        return
    elif 'subtract_without_borrow' in expression:
        p, q = [evaluate_expression(arg.strip(), context) for arg in expression[len('subtract_without_borrow('):-1].split(',')]
        value = subtract_without_borrow(p, q)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name in var_names:
            context[var_name] = value
        return
    elif 'add_tenth_place' in expression:
        p, q = [evaluate_expression(arg.strip(), context) for arg in expression[len('add_tenth_place('):-1].split(',')]
        value = add_tenth_place(p, q)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name in var_names:
            context[var_name] = value
        return
    elif 'subtract_tenth_place' in expression:
        p, q = [evaluate_expression(arg.strip(), context) for arg in expression[len('subtract_tenth_place('):-1].split(',')]
        value = subtract_tenth_place(p, q)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name in var_names:
            context[var_name] = value
        return
    elif 'generate_quotient_rem' in expression:
            args = [arg.strip() for arg in expression[len('generate_quotient_rem('):-1].split(',')]
            min_a = evaluate_expression(args[0], context)
            max_a = evaluate_expression(args[1], context)
            divisors_str = ','.join(args[2:]).strip('[]')
            divisors = [evaluate_expression(d.strip(), context) for d in divisors_str.split(',')]
            values = generate_quotient_rem(min_a, max_a, divisors)
            if values:
                context["a"] = values["a"]
                context["b"] = values["b"]
                context["correct"] = [values["quotient"]]
            return
    elif 'generate_perc_values(' in expression:
        args = [arg.strip() for arg in expression[len('generate_perc_values('):-1].split(',')]
        x_start = evaluate_expression(args[0].strip(), context)
        x_end = evaluate_expression(args[1].strip(), context)
        x_step = evaluate_expression(args[2].strip(), context)
        a_start = evaluate_expression(args[3].strip(), context)
        a_end = evaluate_expression(args[4].strip(), context)
        a_step = evaluate_expression(args[5].strip(), context)
        x_exclude = evaluate_expression(args[6].strip(), context) if len(args) > 6 and args[6].strip() else 0
        a_exclude = evaluate_expression(args[7].strip(), context) if len(args) > 7 and args[7].strip() else 0
        
        x, z, a = generate_perc_values(x_start, x_end, x_step, a_start, a_end, a_step, x_exclude, a_exclude)   
        var_pairs = [var.strip() for var in var_names.strip().strip("()").split(',')]
        context[var_pairs[0]] = x
        context[var_pairs[1]] = z
        context['correct'] = [a]  # Ensure 'a' is wrapped in a list
        return
    elif 'generate_perc_factor_values(' in expression:
        args = [arg.strip() for arg in expression[len('generate_perc_factor_values('):-1].split(',')]
        y_start = evaluate_expression(args[0].strip(), context)
        y_end = evaluate_expression(args[1].strip(), context)
        y_step = evaluate_expression(args[2].strip(), context)
        a_start = evaluate_expression(args[3].strip(), context)
        a_end = evaluate_expression(args[4].strip(), context)
        a_step = evaluate_expression(args[5].strip(), context)
        y_exclude = evaluate_expression(args[6].strip(), context) if len(args) > 6 and args[6].strip() else 0
        a_exclude = evaluate_expression(args[7].strip(), context) if len(args) > 7 and args[7].strip() else 0
        
        y, z, a = generate_perc_factor_values(y_start, y_end, y_step, a_start, a_end, a_step, y_exclude, a_exclude)
        var_pairs = [var.strip() for var in var_names.strip().strip("()").split(',')]
        context[var_pairs[0]] = y
        context[var_pairs[1]] = z
        context[var_pairs[2]] = a
        context['correct'] = [a]  # Ensure 'a' is wrapped in a list
        return
    elif 'generate_xy_values(' in expression:
        args = [arg.strip() for arg in expression[len('generate_xy_values('):-1].split(',')]
        x_start = evaluate_expression(args[0].strip(), context)
        x_end = evaluate_expression(args[1].strip(), context)
        x_step = evaluate_expression(args[2].strip(), context)
        y_start = evaluate_expression(args[3].strip(), context)
        y_end = evaluate_expression(args[4].strip(), context)
        y_step = evaluate_expression(args[5].strip(), context)
        x_exclude = evaluate_expression(args[6].strip(), context) if len(args) > 6 and args[6].strip() else 0
        y_exclude = evaluate_expression(args[7].strip(), context) if len(args) > 7 and args[7].strip() else 0
        
        x, y = generate_xy_values(x_start, x_end, x_step, y_start, y_end, y_step, x_exclude, y_exclude)
        var_pairs = [var.strip() for var in var_names.strip().strip("()").split(',')]
        context[var_pairs[0]] = x
        context[var_pairs[1]] = y
        return
    elif 'generate_perc_alternate_mul(' in expression:
            args = [arg.strip() for arg in expression[len('generate_perc_alternate_mul('):-1].split(',')]
            min_a = int(evaluate_expression(args[0].strip(), context))
            max_a = int(evaluate_expression(args[1].strip(), context))
            min_x = int(evaluate_expression(args[2].strip(), context))
            max_x = int(evaluate_expression(args[3].strip(), context))

            values = generate_perc_alternate_mul(min_a, max_a, min_x, max_x)
            
            context["a"] = values["a"]
            context["x"] = values["x"]
            return
    
    elif 'generate_ratios_a_constant(' in expression:
        args = [arg.strip() for arg in expression[len('generate_ratios_a_constant('):-1].split(',')]
        a_range_start = int(evaluate_expression(args[0].strip(), context))
        a_range_end = int(evaluate_expression(args[1].strip(), context))
        b_range_start = int(evaluate_expression(args[2].strip(), context))
        b_range_end = int(evaluate_expression(args[3].strip(), context))

        values = generate_ratios_a_constant(a_range_start, a_range_end, b_range_start, b_range_end)

        if values:
            for i in range(4):
                context[f"a{i+1}"] = values[f"a{i+1}"]
                context[f"b{i+1}"] = values[f"b{i+1}"]
            context["options"] = values["options"]
        return

    elif 'generate_ratios_b_constant(' in expression:
        args = [arg.strip() for arg in expression[len('generate_ratios_b_constant('):-1].split(',')]
        a_range_start = int(evaluate_expression(args[0].strip(), context))
        a_range_end = int(evaluate_expression(args[1].strip(), context))
        b_range_start = int(evaluate_expression(args[2].strip(), context))
        b_range_end = int(evaluate_expression(args[3].strip(), context))

        values = generate_ratios_b_constant(a_range_start, a_range_end, b_range_start, b_range_end)

        if values:
            for i in range(4):
                context[f"a{i+1}"] = values[f"a{i+1}"]
                context[f"b{i+1}"] = values[f"b{i+1}"]
            context["options"] = values["options"]
        return
    
    elif 'generate_ratios_divisors(' in expression:        
        values = generate_ratios_divisors()
        if values:
            for i in range(4):
                context[f"a{i+1}"] = values[f"a{i+1}"]
                context[f"b{i+1}"] = values[f"b{i+1}"]
            context["options"] = values["options"]
        return
    elif 'generate_ratios_with_denoms(' in expression:
        args = [arg.strip() for arg in expression[len('generate_ratios_with_denoms('):-1].split(',')]
        min_range = int(evaluate_expression(args[0].strip(), context))
        max_range = int(evaluate_expression(args[1].strip(), context))

        values = generate_ratios_with_denoms(min_range, max_range)

        if values:
            for i in range(4):
                context[f"a{i+1}"] = values[f"a{i+1}"]
                context[f"b{i+1}"] = values[f"b{i+1}"]
            context["options"] = values["options"]
        return

    elif 'generate_ratios_reduced(' in expression:
        values = generate_ratios_reduced()

        context["ratio_1"] = values["ratio_1"]
        context["ratio_2"] = values["ratio_2"]
        context["correct"] = [values["is_equivalent"]]
        return
    elif 'generate_ratios_equivalent(' in expression:
        values = generate_ratios_equivalent()

        context["ratio_1"] = values["ratio_1"]
        context["ratio_2"] = values["ratio_2"]
        context["correct"] = [values["is_equivalent"]]
        return
    elif 'generate_abs_value_question(' in expression:
        values = generate_abs_value_question()
        
        if values:
            context["equation"] = values["equation"]
            context["is_true"] = values["is_true"]
            context["correct"] = values["correct"]
            context["x"] = values["x"]
            context["y"] = values["y"]
        return
    elif 'generate_integer_subtraction_question(' in expression:
        values = generate_integer_subtraction_question()
        
        if values:
            context["equation"] = values["equation"]
            context["is_true"] = values["is_true"]
            context["correct"] = values["correct"]
            context["p"] = values["p"]
            context["q"] = values["q"]
            context["y"] = values["y"]
        return
    elif 'generate_integer_addition_question(' in expression:
        values = generate_integer_addition_question()
        
        if values:
            context["equation"] = values["equation"]
            context["is_true"] = values["is_true"]
            context["correct"] = values["correct"]
            context["p"] = values["p"]
            context["q"] = values["q"]
            context["y"] = values["y"]
        return
    elif 'generate_ratios_not_reduced(' in expression:
        args = [arg.strip() for arg in expression[len('generate_ratios_not_reduced('):-1].split(',')]
        min_val = int(evaluate_expression(args[0].strip(), context))
        max_val = int(evaluate_expression(args[1].strip(), context))

        values = generate_ratios_not_reduced(min_val, max_val)

        context["ratio_1"] = values["ratio_1"]
        context["ratio_2"] = values["ratio_2"]
        context["correct"] = [values["is_equivalent"]]
        return
    elif 'generate_ratios_find_value(' in expression:
        values = generate_ratios_find_value()
        context["ratio_1"] = values["ratio_1"]
        context["ratio_2"] = values["ratio_2"]
        context["correct"] = [values["correct_answer"]]
        context["incorrect"] = values["options"]
        return
    elif 'generate_ratios_find_reduced(' in expression:
        values = generate_ratios_find_reduced()
        context["ratio_1"] = values["ratio_1"]
        context["ratio_2"] = values["ratio_2"]
        context["correct"] = [values["correct_answer"]]
        context["incorrect"] = values["options"]
        return
    elif 'generate_ratios_find_not_reduced(' in expression:
        args = [arg.strip() for arg in expression[len('generate_ratios_find_not_reduced('):-1].split(',')]
        min_val = int(evaluate_expression(args[0].strip(), context))
        max_val = int(evaluate_expression(args[1].strip(), context))

        values = generate_ratios_find_not_reduced(min_val, max_val)

        context["ratio_1"] = values["ratio_1"]
        context["ratio_2"] = values["ratio_2"]
        context["correct"] = [values["correct_answer"]]
        context["incorrect"] = values["options"]
        return
    elif expression.startswith('closest_sum'):
        d, e = [evaluate_expression(arg.strip(), context) for arg in expression[len('closest_sum('):-1].split(',')]
        values = closest_sum(d, e)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, values):
            context[var_name] = val
        return
    elif expression.startswith('closest_diff'):
        d, e = [evaluate_expression(arg.strip(), context) for arg in expression[len('closest_diff('):-1].split(',')]
        values = closest_diff2(d, e)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, values):
            context[var_name] = val
        return
    elif expression.startswith('min('):
        args = [ arg.strip() for arg in expression[len('min('):-1].split(',') ]
        # print("ARGS", args, flush=True)
        values = [evaluate_expression(arg.strip(), context) for arg in expression[len('min('):-1].split(',')]
        # print("VALUES", values, flush=True)
        value = min(values)
        # print("VALUE", value, flush=True)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name in var_names:
            context[var_name] = value
        return
    elif expression.startswith('generate_sum_questions('):
        args = [arg.strip() for arg in expression[len('generate_sum_questions('):-1].split(',')]
        is_carry = args[0].strip().lower() == 'true'
        args = args[1:]
        min_n, max_n, min_a, max_a, min_b, max_b, num_options = \
        [evaluate_expression(arg.strip(), context) for arg in args]
        values = generate_sum_questions(is_carry, min_n, max_n, min_a, max_a, min_b, max_b, num_options)
        context["correct"] = values["correct_answers"]
        context["options"] = values["options"]
        context["options_length"] = num_options
        sum = values["sum"]
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name in var_names:
            context[var_name] = sum
        return
    elif expression.startswith('generate_diff_questions('):
        args = [arg.strip() for arg in expression[len('generate_diff_questions('):-1].split(',')]
        is_borrow = args[0].strip().lower() == 'true'
        args = args[1:]
        min_d, max_d, min_a, max_a, min_b, max_b, num_options = \
            [evaluate_expression(arg.strip(), context) for arg in args]
        values = generate_diff_questions(is_borrow, min_d, max_d, min_a, max_a, min_b, max_b, num_options)
        context["correct"] = values["correct_answers"]
        context["options"] = values["options"]
        context["options_length"] = num_options
        diff = values["diff"]
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name in var_names:
            context[var_name] = diff
        return
    elif expression.startswith('generate_mul_questions('):
        args = [arg.strip() for arg in expression[len('generate_mul_questions('):-1].split(',')]
        is_borrow = args[0].strip().lower() == 'true'
        args = args[1:]
        min_x, max_x, min_y, max_y, num_options = \
            [evaluate_expression(arg.strip(), context) for arg in args]
        values = generate_mul_questions(is_borrow, min_x, max_x, min_y, max_y, num_options)
        if values:
            context["correct"] = values["correct_answers"]
            context["options"] = values["options"]
            context["options_length"] = num_options
            product = values["product"]
            var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
            for var_name in var_names:
                context[var_name] = product
        return
    elif expression.startswith('generate_div_quotient('):
        args = [arg.strip() for arg in expression[len('generate_div_quotient('):-1].split(',')]
        min_a = evaluate_expression(args[0], context)
        max_a = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context)
        divisors_str = ','.join(args[3:]).strip('[]')
        divisors = [evaluate_expression(d.strip(), context) for d in divisors_str.split(',')]
        values = generate_div_quotient(min_a, max_a, num_options, divisors)
        if values:
            context["correct"] = values["correct_answers"]
            context["options"] = values["options"]
            context["options_length"] = num_options
            quotient = values["quotient"]
            var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
            for var_name in var_names:
                context[var_name] = quotient
        return
    elif expression.startswith('generate_div_quotient_with_remainder('):
        print("*******")
        print(expression[len('generate_div_quotient_with_remainder('):-1].split(','))
        args = [arg.strip() for arg in expression[len('generate_div_quotient_with_remainder('):-1].split(',')]
        min_a = evaluate_expression(args[0], context)
        max_a = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context)
        divisors_str = ','.join(args[3:]).strip('[]')
        divisors = [evaluate_expression(d.strip(), context) for d in divisors_str.split(',')]
        values = generate_div_quotient_with_remainder(min_a, max_a, num_options, divisors)
        if values:
            context["correct"] = values["correct_answers"]
            context["options"] = values["options"]
            context["options_length"] = num_options
            quotient = values["quotient"]
            var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
            for var_name in var_names:
                context[var_name] = quotient
        return
    elif expression.startswith('generate_div_options_remainder('):
        args = [arg.strip() for arg in expression[len('generate_div_options_remainder('):-1].split(',')]
        min_a = evaluate_expression(args[0], context)
        max_a = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context)
        d = evaluate_expression(args[3], context)
        divisors_str = ','.join(args[4:])  # Join the remaining parts correctly

        # Process the divisors list
        divisors = [int(d.strip()) for d in re.findall(r'\d+', divisors_str)]
        values = generate_div_options_remainder(min_a, max_a, num_options, d, divisors)

        if values:
            for i in range(num_options):
                context[f"a{i+1}"] = values[f"a{i+1}"]
                context[f"b{i+1}"] = values[f"b{i+1}"]
            context["options"] = values["options"]
        return
    elif expression.startswith('generate_remainder_options('):
        args = [arg.strip() for arg in expression[len('generate_remainder_options('):-1].split(',')]
        min_a = evaluate_expression(args[0], context)
        max_a = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context)
        divisors_str = ','.join(args[3:]).strip('[]')
        divisors = [evaluate_expression(d.strip(), context) for d in divisors_str.split(',')]
        values = generate_remainder_options(min_a, max_a, num_options, divisors)
        if values:
            context["correct"] = values["correct_answers"]
            context["options"] = values["options"]
            context["options_length"] = num_options
            remainder = values["remainder"]
            var_names_list = [var.strip() for var in var_names.strip().strip("()").split(',')]
            for var_name in var_names_list:
                context[var_name] = remainder
        return
    elif 'generate_with_carry_options(' in expression:
        args = [arg.strip() for arg in expression[len('generate_with_carry_options('):-1].split(',')]
        min_x = evaluate_expression(args[0].strip(), context)
        max_x = evaluate_expression(args[1].strip(), context)
        min_y = evaluate_expression(args[2].strip(), context)
        max_y = evaluate_expression(args[3].strip(), context)
        num_options = evaluate_expression(args[4].strip(), context) if len(args) > 4 else 4
        d = evaluate_expression(args[5].strip(), context) if len(args) > 5 else 50

        values = generate_with_carry_options(min_x, max_x, min_y, max_y, num_options, d)

        if values:
            for i in range(num_options):
                context[f"a{i+1}"] = values[f"a{i+1}"]
                context[f"b{i+1}"] = values[f"b{i+1}"]
            context["options"] = values["options"]
        return
    elif 'generate_without_carry_options(' in expression:
        args = [arg.strip() for arg in expression[len('generate_without_carry_options('):-1].split(',')]
        min_x = evaluate_expression(args[0].strip(), context)
        max_x = evaluate_expression(args[1].strip(), context)
        min_y = evaluate_expression(args[2].strip(), context)
        max_y = evaluate_expression(args[3].strip(), context)
        num_options = evaluate_expression(args[4].strip(), context) if len(args) > 4 else 4
        d = evaluate_expression(args[5].strip(), context) if len(args) > 5 else 50

        values = generate_without_carry_options(min_x, max_x, min_y, max_y, num_options, d)

        if values:
            for i in range(num_options):
                context[f"a{i+1}"] = values[f"a{i+1}"]
                context[f"b{i+1}"] = values[f"b{i+1}"]
            context["options"] = values["options"]
        return
    elif expression.startswith('generate_div_options('):
        args = [arg.strip() for arg in expression[len('generate_div_options('):-1].split(',')]
        min_a = evaluate_expression(args[0], context)
        max_a = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context)
        d = evaluate_expression(args[3], context)
        divisors_str = ','.join(args[4:])  # Join the remaining parts correctly

        # Process the divisors list
        divisors = [int(d.strip()) for d in re.findall(r'\d+', divisors_str)]
        values = generate_div_options(min_a, max_a, num_options, d, divisors)

        if values:
            for i in range(num_options):
                context[f"a{i+1}"] = values[f"a{i+1}"]
                context[f"b{i+1}"] = values[f"b{i+1}"]
            context["options"] = values["options"]
        return
    elif 'set_clock_delta(' in expression:
        args = expression[len('set_clock_delta('):-1].strip().split(',')
        x_list = [evaluate_expression(arg.strip(), context) for arg in args[0].strip().strip('[]').split(',')]
        y_list = [evaluate_expression(arg.strip(), context) for arg in args[1].strip().strip('[]').split(',')]
        delta_info = set_clock_delta(x_list, y_list)
        var_name = var_names.strip().strip("()")
        context.update(delta_info)  # Update context with all returned values
        context['correct'] = [delta_info['result_time']]  # Set result_time as the correct answer list
        context['question_value'] = delta_info['result_time']         # context['h1'] = [delta_info['h1']]
        return
    
    elif 'generate_wrong_multiples' in expression:
        args = [arg.strip() for arg in expression[len('generate_wrong_multiples('):-1].split(',')]
        X = evaluate_expression(args[0], context)
        num_multiples = evaluate_expression(args[1], context) if len(args) > 1 else 9
        values = generate_wrong_multiples(X, num_multiples)
        if values:
            for i in range(num_multiples):
                context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        return
    
    elif 'generate_close_negatives(' in expression:
        args = [arg.strip() for arg in expression[len('generate_close_negatives('):-1].split(',')]
        min_range = evaluate_expression(args[0], context)
        max_range = evaluate_expression(args[1], context)
        max_difference = evaluate_expression(args[2], context) if len(args) > 2 else 10

        values = generate_close_negatives(min_range, max_range, max_difference)

        for i in range(len(values)):
            context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        context["options"] = values
        return

    elif 'generate_mixed_numbers(' in expression:
        args = [arg.strip() for arg in expression[len('generate_mixed_numbers('):-1].split(',')]
        min_range = evaluate_expression(args[0], context)
        max_range = evaluate_expression(args[1], context)
        max_difference = evaluate_expression(args[2], context) if len(args) > 2 else 10

        values = generate_mixed_numbers(min_range, max_range, max_difference)

        for i in range(len(values)):
            context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        context["options"] = values
        return

    elif 'generate_abs_mixed_numbers(' in expression:
        args = [arg.strip() for arg in expression[len('generate_abs_mixed_numbers('):-1].split(',')]
        min_range = evaluate_expression(args[0], context)
        max_range = evaluate_expression(args[1], context)
        max_difference = evaluate_expression(args[2], context) if len(args) > 2 else 10

        values = generate_abs_mixed_numbers(min_range, max_range, max_difference)

        for i in range(len(values)):
            context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        context["options"] = values
        return
    
    elif 'generate_pos_mixed_numbers(' in expression:
        args = [arg.strip() for arg in expression[len('generate_pos_mixed_numbers('):-1].split(',')]
        min_range = evaluate_expression(args[0], context)
        max_range = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context) if len(args) > 2 else 4

        values = generate_pos_mixed_numbers(min_range, max_range, num_options)

        for i in range(len(values)):
            context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        context["options"] = values
        return
    
    elif 'generate_neg_mixed_numbers(' in expression:
        args = [arg.strip() for arg in expression[len('generate_neg_mixed_numbers('):-1].split(',')]
        min_range = evaluate_expression(args[0], context)
        max_range = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context) if len(args) > 2 else 4

        values = generate_neg_mixed_numbers(min_range, max_range, num_options)

        for i in range(len(values)):
            context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        context["options"] = values
        return
    
    elif 'generate_mixed_numbers_new(' in expression:
        args = [arg.strip() for arg in expression[len('generate_mixed_numbers_new('):-1].split(',')]
        min_range = evaluate_expression(args[0], context)
        max_range = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context) if len(args) > 2 else 4

        values = generate_mixed_numbers_new(min_range, max_range, num_options)

        for i in range(len(values)):
            context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        context["options"] = values
        return
    
    elif 'generate_mixed_numbers_special(' in expression:
        args = [arg.strip() for arg in expression[len('generate_mixed_numbers_special('):-1].split(',')]
        min_range = evaluate_expression(args[0], context)
        max_range = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context) if len(args) > 2 else 4

        values = generate_mixed_numbers_special(min_range, max_range, num_options)

        for i in range(len(values)):
            context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        context["options"] = values
        return

    elif 'generate_addition_abs_mixed_numbers' in expression:
        values = generate_addition_abs_mixed_numbers()

        for i in range(len(values)):
            context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        context["options"] = values
        return

    elif 'generate_subtraction_abs_mixed_numbers' in expression:
        values = generate_subtraction_abs_mixed_numbers()

        for i in range(len(values)):
            context[chr(97 + i)] = values[i]  # 97 is the ASCII code for 'a'
        context["options"] = values
        return

    elif expression.startswith('generate_equivalent_fractions('):
        num_options = 4
        args = [arg.strip() for arg in expression[len('generate_equivalent_fractions('):-1].split(',')]
        multiplier_type = args[-1]  # Get the last argument
        #remove the last argument
        args = args[:-1]
        min_p, max_p, min_q, max_q, min_multiplier, max_multiplier = [evaluate_expression(arg.strip(), context) for arg in args]
        values = generate_equivalent_fractions(min_p, max_p, min_q, max_q, min_multiplier, max_multiplier, multiplier_type, num_options )
        # print("VALUES", values, flush=True)
        context["correct"] = values["correct_answers"]
        context["options"] = values["options"]
        context["options_length"] = num_options
        numerator = values["numerator"]
        denominator = values["denominator"]
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, (numerator, denominator)):
            context[var_name] = val
        return
    elif expression.startswith('equi_frac_number_line('):
        max_partitions = 15
        p_list = [2, 2, 3, 3]
        q_list = [3, 5, 4, 5]
        i = random.randint(0, 3)
        p = p_list[i]
        q = q_list[i]
        context["start_point"] = [0]
        context["end_point"] = [1]
        a = random.randint(2,3)
        b = a+1
        mul_numerator_drop = p*a
        mul_denominator_drop = q*a
        mul_numerator_number_line = p*b
        mul_denominator_number_line = q*b
        if mul_denominator_number_line > max_partitions:
            op1 = Fraction(mul_numerator_number_line, mul_denominator_number_line)
            op2 = Fraction(mul_numerator_drop, mul_denominator_drop)
            interval_length = mul_denominator_drop
            question_numerator = mul_numerator_number_line
            question_denominator = mul_denominator_number_line
        else:
            op1 = Fraction(mul_numerator_drop, mul_denominator_drop)
            op2 = Fraction(mul_numerator_number_line, mul_denominator_number_line)
            interval_length = mul_denominator_number_line
            question_numerator = mul_numerator_drop
            question_denominator = mul_denominator_drop

            if random.choice([True, False]):
                op1, op2 = op2, op1
                interval_length = mul_denominator_drop
                question_numerator = mul_numerator_number_line
                question_denominator = mul_denominator_number_line

        correct_answer = [op2]
        context["correct"] = correct_answer
        context["interval_length"] = Fraction(1, interval_length)
        var_names = [var.strip() for var in var_names.strip().strip("()").split(',')]
        for var_name, val in zip(var_names, (question_numerator, question_denominator, interval_length, p, q)):
            context[var_name] = val
        return
    elif 'generate_mul_options2(' in expression:
        args = [arg.strip() for arg in expression[len('generate_mul_options2('):-1].split(',')]
        min_x = evaluate_expression(args[0].strip(), context)
        max_x = evaluate_expression(args[1].strip(), context)
        digits_y = [evaluate_expression(arg.strip(), context) for arg in args[2].strip().strip('[]').split(',') if arg.strip()]
        num_options = evaluate_expression(args[3].strip(), context)
        unique = evaluate_expression(args[4].strip(), context) if len(args) > 4 else False
        
        values = generate_mul_options2(min_x, max_x, digits_y, num_options, unique)
        
        if values:
            var_pairs = [var.strip() for var in var_names.strip().strip("()").split(',')]
            var_a, var_b = var_pairs
            
            # Assuming the first option is the correct answer to extract variables a and b
            a, b = map(int, values['options'][0].split(' x '))
            
            context[var_a] = a
            context[var_b] = b
        
        return
    
    elif 'generate_mul_options_single(' in expression:
        args = [arg.strip() for arg in expression[len('generate_mul_options_single('):-1].split(',')]
        min_a = evaluate_expression(args[0], context)
        max_a = evaluate_expression(args[1], context)
        num_options = evaluate_expression(args[2], context)
        digits_y_str = ','.join(args[3:])  # Join the remaining parts correctly

        # Process the digits_y list
        digits_y = [int(d.strip()) for d in re.findall(r'\d+', digits_y_str)]

        # Extract and evaluate the last argument (d)
        t = evaluate_expression(args[-1], context)

        # Call the function with all necessary arguments
        values = generate_mul_options_single(min_a, max_a, num_options, digits_y, t)

        if values:
            for i in range(num_options):
                context[f"a{i+1}"] = values[f"a{i+1}"]
                context[f"b{i+1}"] = values[f"b{i+1}"]
            context["options"] = values["options"]
        return
    
    elif 'generate_addition_option(' in expression:
        args = [arg.strip() for arg in expression[len('generate_addition_option('):-1].split(',')]
        min_value = evaluate_expression(args[0], context)
        max_value = evaluate_expression(args[1], context)
        
        # Call the function with all necessary arguments
        values = generate_addition_option(min_value, max_value)
        
        if values:
            context["p"] = values["p"]
            context["q"] = values["q"]
            context["s"] = values["s"]
            context["c"] = values["c"]
            context["option"] = values["option"]
        return
    
    elif 'generate_addition_option_with_constraints(' in expression:
        args = [arg.strip() for arg in expression[len('generate_addition_option_with_constraints('):-1].split(',')]
        min_p = evaluate_expression(args[0], context)
        max_p = evaluate_expression(args[1], context)
        min_q = evaluate_expression(args[2], context)
        max_q = evaluate_expression(args[3], context)
        
        # Call the function with all necessary arguments
        values = generate_addition_option_with_constraints(min_p, max_p, min_q, max_q)
        
        if values:
            context["p"] = values["p"]
            context["q"] = values["q"]
            context["s"] = values["s"]
            context["c"] = values["c"]
            context["option"] = values["option"]
        return
    
    elif 'generate_addition_option_with_opposite_constraints(' in expression:
        args = [arg.strip() for arg in expression[len('generate_addition_option_with_opposite_constraints('):-1].split(',')]
        min_p = evaluate_expression(args[0], context)
        max_p = evaluate_expression(args[1], context)
        min_q = evaluate_expression(args[2], context)
        max_q = evaluate_expression(args[3], context)
        
        # Call the function with all necessary arguments
        values = generate_addition_option_with_opposite_constraints(min_p, max_p, min_q, max_q)
        
        if values:
            context["p"] = values["p"]
            context["q"] = values["q"]
            context["s"] = values["s"]
            context["c"] = values["c"]
            context["option"] = values["option"]
        return



    var_names = [var.strip() for var in var_names.split(',')]

    if expression.startswith('no_borrow('):
        x, y = [evaluate_expression(arg.strip(), context) for arg in expression[len('no_borrow('):-1].split(',')]
        value = subtract_without_borrow(x, y)
        for var_name in var_names:
            context[var_name] = value
    elif expression.startswith('rev('):
        # Extract arguments within the parentheses
        args = expression[len('rev('):-1]
        # Split arguments by commas, taking care not to split within nested functions or quotes
        args_list = [arg.strip() for arg in args.split(',')]
        
        if len(args_list) == 1:
            x = evaluate_expression(args_list[0], context)
            for var_name in var_names:
                context[var_name] = rev(x)
        elif len(args_list) == 2:
            x = evaluate_expression(args_list[0], context)
            y = evaluate_expression(args_list[1], context)
            for var_name in var_names:
                context[var_name] = rev(x,y)
    elif expression.startswith('divisor('):
        y = evaluate_expression(expression[len('divisor('):-1], context)
        value = divisor(y)
        for var_name in var_names:
            context[var_name] = value
    elif expression.startswith('smallest_divisor('):
        y = evaluate_expression(expression[len('smallest_divisor('):-1], context)
        value = smallest_divisor(y)
        for var_name in var_names:
            context[var_name] = value
    elif expression.startswith('abs_value('):
        y = evaluate_expression(expression[len('abs_value('):-1], context)
        value = abs_value(y)
        for var_name in var_names:
            context[var_name] = value
    elif expression.startswith('floor('):
        value = evaluate_expression(expression[len('floor('):-1], context)
        floor_float = math.floor(eval(str(value)))
        floor_int = int(floor_float)
        for var_name in var_names:
            context[var_name] = floor_int
    elif expression.startswith('ceil('):
        value = evaluate_expression(expression[len('ceil('):-1], context)
        ceil_floor = math.ceil(eval(str(value)))
        ceil_int = int(ceil_floor)
        for var_name in var_names:
            context[var_name] = ceil_int
    
    # Check if the line contains a recognized keyword and process accordingly
    elif var_names[0] in recognized_keywords:
        # If it's a keyword, evaluate the expression and store the result under the keyword
        context[var_names[0]] = evaluate_expression(expression, context)
    else:
        # Process as variable assignment
        if expression.isdigit():  # Direct number assignment as integer
            value = int(expression)
            value_generator = lambda: value
        if 'select(' in expression:
            # print("SELECT EXPRESSION", expression, flush=True)
            values = expression[len('select('):-1].split(',')
            values = [evaluate_expression(val.strip(), context) for val in values]
            condition = values[0]
            value_generator = lambda: values[1] if condition else values[2]
        elif 'range' in expression:
            # print("range EXPRESSION", expression, flush=True)
            value_generator = parse_range(expression, context)
        elif '[' in expression:  # Evaluate list of expressions or numbers
            values = [evaluate_expression(val.strip(), context) for val in expression[1:-1].split(',')]
            value_generator = lambda: random.choice(values)
        else:
            value = evaluate_expression(expression, context)  # Evaluate any other expression
            value_generator = lambda: value

        for var_name in var_names:
            context[var_name] = value_generator()

def get_value_from_context(key, context):
    possible_delimiters = ['+', '-']
    current_delimiter = '-'
    for delimiter in possible_delimiters:
        if delimiter in key:
            current_delimiter = delimiter
            break
    parts = key.split(current_delimiter)
    # print("KEY", key, flush=True)
    # print("DELIMITER", current_delimiter, flush=True)
    # print("PARTS", parts, flush=True)
    # print("CONTEXT", context, flush=True)
    values = []
    for part in parts:
        part = part.strip()  # Remove any leading or trailing whitespace
        if part in context:
            values.append(str(context[part]))
        else:
            values.append(part)  # Keep the original part if not found
    return current_delimiter.join(values)

def parse_array_results(line, context, result_type, key=None):
    # print("LINE", line, flush=True)
    # print("CONTEXT", context, flush=True)
    # print("RESULT TYPE", result_type, flush=True)
    # print("KEY", key, flush=True)
    var_name, expression = line.split('=', 1)
    var_name = var_name.strip()
    expressions = expression.strip()[1:-1].split(',')
    # print("EXPRESSIONS", expressions, flush=True)
    if key == "options" and context.get("options_type", "") == "no_eval":
        expressions = [get_value_from_context(expression, context) for expression in expressions]
        context[var_name] = expressions
        return
    results = [evaluate_expression(exp.strip(), context) for exp in expressions]
    context[var_name] = results

# def parse_pseudo_code(pseudo_code):
#     context = {}
#     result_types = {}
#     lines = pseudo_code.strip().split('\n')
    
#     for line in lines:
#         line = line.strip()
#         print(line)
#         if 'variables' in line or 'correct_type' in line or 'incorrect_type' in line:
#             key, value = line.split('=')
#             result_types[key.strip().replace('_type', '')] = value.strip().lower()
#         else:
#             if '=' in line:
#                 if 'correct' in line or 'incorrect' in line:
#                     parse_array_results(line, context, result_types[line.split('=')[0].strip()])
#                 else:
#                     parse_assignment(line, context)
    
#     return context

def parse_pseudo_code(pseudo_code):
    context = {}
    # print("PSEUDO CODE", pseudo_code, flush=True)
    lines = pseudo_code.strip().split('\n')
    
    for line in lines:
        # continue if empty line
        if not line.strip():
            continue
        # print(line, flush=True)
        # print("CONTEXT", context, flush=True)
        if 'incorrect_value' in line:
            if '<' in line:
                key, value = line.split('<')
                context["max_incorrect_value"] = value.strip()
            elif '>' in line:
                key, value = line.split('>')
                context["min_incorrect_value"] = value.strip()
            continue
        line = line.strip()
        # print("after stripping line", line, flush=True)
        key, value = line.split('=', 1)  # Split only on the first '=', ensuring proper key/value separation
        key = key.strip()
        value = value.strip()

        if key in recognized_keywords:
            # Direct handling of recognized keywords
            context[key] = value
        elif "interval_length" in line:
            # if interval_length is a fraction, store it as a Fraction object
            interval_length = evaluate_expression(value, context)
            context["interval_length"] = interval_length
        elif "question_value" in line:
            # if interval_length is a fraction, store it as a Fraction object
            context["question_value"] = evaluate_expression(value, context)
        elif 'variables' in line or 'correct_type' in line or 'incorrect_type' in line:
            key, value = line.split('=')
            context[key.strip().replace('_type', '')] = value.strip().lower()
        elif 'incorrect' in line and 'select' in line:
            print("CONTEXT", context, flush=True)
            parse_assignment(line, context)
        elif key in ['correct', 'incorrect', 'options', 'start_point', 'end_point', 'incorrect_1', 'incorrect_2', 'error_margin']:
            # Handle arrays specifically for 'correct' or 'incorrect' keywords
            parse_array_results(line, context, context.get(key, 'decimal'), key)  # Default type can be adjusted
        elif 'incorrect_static' in line:
            # Extract incorrect_static value
            key, value = line.split('=')
            context['incorrect_static'] = evaluate_expression(value.strip(), context)
        elif 'incorrect_dynamic' in line:
            # Extract incorrect_dynamic value
            key, value = line.split('=')
            context['incorrect_dynamic'] = evaluate_expression(value.strip(), context)
        elif 'positive_incorrect' in line:
            # Extract positive_incorrect value
            key, value = line.split('=')
            context['positive_incorrect'] = int(value.strip())
        elif 'negative_incorrect' in line:
            # Extract negative_incorrect value
            key, value = line.split('=')
            context['negative_incorrect'] = int(value.strip())
        elif 'set_clock_delta(' in value:
            parse_assignment(line, context)
        else:
            # Handle general assignment
            parse_assignment(line, context)
    
    return context


"""
    The Following code is to Generate the Questions Data

    1) Copy the whole code in Question Manager (except the import statements) and run it in ipython
    2) Update the question_text and rules data below
    3) Run the following updated code in ipython
    4) data = generate_data()
"""

class QuestionType:
    SINGLE_CHOICE = "SINGLE_CHOICE"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    ORDER_SEQUENCE = "ORDER_SEQUENCE"
    MATCHING = "MATCHING"
    FILL_IN_THE_BLANK = "FILL_IN_THE_BLANK"

def generate_data(question_text="", rules=""""""):
    if not question_text:
        question_text = "Set the time to {{x}}:{{y}}."
    if not rules:
        rules = """
            variables = [x,y,p]
            x = range(1,12)
            y = [0,30]
            p = select(y==0, 6, 3)
            question_value = [60x+y]
            correct = [60x+y]
            error_margin = [p,p,3,3]
            question = input_clock_1
        """
    
    class QuestionTemplate:
        def __init__(self, question_text, q_rules, a_rules, extra_rules):
            self.question_text = {
                "question_text" : question_text
            }
            self.question_constraints = {
                "q_rules": q_rules,
                "extra_rules": extra_rules
            }
            self.answer_constraints = {
                "a_rules": a_rules
            }
    question_template = QuestionTemplate(question_text, rules,"","")
    return QuestionManager.generate_questions(20, question_template)

def generate_data_helper():
    question_text = "Arrange from <smallest/largest> to <largest/smallest>"
    rules = """
variables = [n]
n = generate_ratios_divisors()
options_length = 4
options_type = no_eval
order = any
    """
    data = generate_data(question_text, rules)
    return data
data = generate_data_helper()

#For testing priority options - incorrect_dynamic - P1, incorrect_static - P2

# def generate_data_helper():
#     question_text = "What's {{a}}+{{b}}+{{c}}+{{d}}"
#     rules = """
# variables = [a,b,c,d,p,q]
# a = range(-50,0)
# b = range(-50,-1)
# c = range(0,50)
# d = range(1,50)
# p = 123456789
# correct = [123456789]
# incorrect_static = [11111, 22222, 333333, 44444, 555555]
# incorrect_dynamic = [-1, -2, 4, 56, 78, 1]
# negative_incorrect = 2
# incorrect_length = 3
#     """
#     data = generate_data(question_text, rules)
#     return data
# data = generate_data_helper()


pseudo_code = """
variables = [a,b,c,d,n]
n = generate_ratios_divisors()
options_length = 4
options_type = no_eval
order = any
"""
print(parse_pseudo_code(pseudo_code))


# The following code shall be used for unit testing

# pseudo_code = """
# variables = [a, b]
# a = -1
# b = a
# incorrect = [a, b]
# """
# print(parse_pseudo_code(pseudo_code))

# pseudo_code = """
# variables = [x]
# x = generate_mixed_numbers_special(-100, 100, 4)
# options_length = 4
# options_type = no_eval
# """
# print(parse_pseudo_code(pseudo_code))


# pseudo_code = """
# variables = [n]
# n = generate_div_quotient_with_remainder(10, 900, 4, [10,100])
# """
# print(parse_pseudo_code(pseudo_code))


# pseudo_code = """
# variables = [a,b]
# b = [10,100]
# a = range(b,900)
# variables = [m,n, i,q,v]
# n = a
# m = b
# i = n%m
# q = floor(n/m)
# v = select(q!=i, q, b+2)
# correct = [i]
# incorrect = [v,(i+1)%b, b+1]
# incorrect_length = 3
# incorrect_value > -1
# """
# print(parse_pseudo_code(pseudo_code))

# pseudo_code = """
# variables = [a, b, c, d, p, q, r, s, options]
# (a, b, c, d, p, q, r, s, options) = generate_div_options(10, 900, 4, [10, 100])
# order= any
# question = number
# answer = number
# """
# print(parse_pseudo_code(pseudo_code))


# pseudo_code = """
# variables = [n]
# n = generate_diff_questions(False, 0, 100,  10, 99, 1, 9, 4)
# question = number
# """
# print(parse_pseudo_code(pseudo_code))

# pseudo_code = """
# variables = [n,x,p,q]
# n = range(2,8)
# x = range(n+1,9)
# p = [x/n, x/(n+1)]
# q = x/(n+2)
# options = [x/n, x/n, p,q]
# correct = [x/n]
# question = fraction
# answer = shape_1
# unique_options = False
# """
# print(parse_pseudo_code(pseudo_code))


# pseudo_code = """
# variables = [m, n , x, y]
# n = range(3,10)
# y = range(2, n)
# m = range(1, n)
# x = range(y, 2y)
# options_length = 4
# options = [0,1,m/n,x/y]
# """
# print(parse_pseudo_code(pseudo_code))


# pseudo_code = """
# variables = [x,y]
# x = range(1,12)
# y = [0,30]
# variables = [a,b,c,d]
# a = range(1,12)
# b = range(0,59)
# d = [25,26,27,28,29,31,32,33,34,35]
# correct = [60*x+y]
# incorrect_1 = [60(x+1)+y, 60(x-1)+y, 12*60+x, 12*60+5x, 60a, 60a+b]
# incorrect_2 = [60(x+1)+y, 60(x-1)+y, 60x, 60(x+1), 60x+6, 6*60+5x, 60x+d, 6a+b]
# incorrect = select(y==0, incorrect_1, incorrect_2)
# incorrect_length = 4
# """
# print(parse_pseudo_code(pseudo_code))


# pseudo_code = """
# variables = [n,m,k,s]
# n = range(2,9)
# m = range(n+1,3n)
# start_point = [0]
# end_point = [m/n]
# interval_length = 1/m
# s = (m+n)/n
# k = range(1, s, 1)
# correct = [k]
# question = number
# answer = fraction
# """
# a = parse_pseudo_code(pseudo_code)
# print(a)


# pseudo_code = """
# variables = [n,p,q,r,s]
# p = range(5, 10)
# q = range(4, p-1)
# r = range(3, q-1)
# s = range(2,r-1)
# options_length = 4
# options=[1/p,1/q,1/r, 1/s]
# order = any
# answer = fraction
# """
# a = parse_pseudo_code(pseudo_code)
# print(a)

# pseudo_code = """
# variables = [p,q]
# (p,q) = generate_equivalent_fractions(1,1,2, 10, 2, 5, integer)
# question = number
# answer = fraction
# """
# a = parse_pseudo_code(pseudo_code)
# print(a)


# pseudo_code = """
# variables = [n,x]
# n = range(2,9)
# x = range(2,n-1)
# correct = [x/n]
# incorrect = [x/(n+1), x/(n+1), x/(n-1)]
# """
# a = parse_pseudo_code(pseudo_code)
# print(a)

# pseudo_code = """
# variables = [x,y, p, q, r, s, t, u]

# (x, y) = no_carry_sum(1, 9, 10, 99)
# (p, q) = no_carry_sum(1, 9, 10, 99)
# (r, s) = no_carry_sum(1, 9, 10, 99)
# (t, u) = no_carry_sum(1, 9, 10, 99)
# options_length = 4

# options = [x+y, p+q, r+s, t+u]
# correct = [x+y, p+q, r+s, t+u]
# """
# print(parse_pseudo_code(pseudo_code))

# pseudo_code = """
# variables = [d,e,n,x,y]
# d = [3,4]
# e = 100
# (n, x, y) = closest_diff(d, e)
# correct = [n]
# question = number
# """
# print(parse_pseudo_code(pseudo_code))

# pseudo_code = """
# variables = [d,e,n,x,y,p,q,r]
# d = [2,3]
# e = 10
# (n, x, y) = closest_sum(d, e)
# p = n/e
# r = min(p,(e-1))
# q = range(0,r)
# start_point = [n-q*e]
# end_point = [n+(e-q-1)*e]
# correct= [n]
# question=number
# """
# print(parse_pseudo_code(pseudo_code))

# pseudo_code = """
# variables = [x,y]
# (x,y) = no_borrow_diff(10,99,1,9)
# variables = [n,r,s,t]
# n = x-y
# r = rev(n)
# s = rev(n-1)
# t = rev(n+1)
# correct= [n]
# incorrect = [n-1, n+1, x+y, r, s, t]
# incorrect_length = 9
# incorrect_value > 0
# question = number
# """
# print(parse_pseudo_code(pseudo_code))



# pseudo_code = """
# variables = [e,f,g]
# (e,f,g) = closest_sum(2, 10)
# variables = [x,y]
# (x, y) = no_carry_sum(1, 9, 10, 99)
# variables = [z,p]
# (z, p) = no_borrow_diff(100, 999, 10, 99)
# variables = [a,b]
# (a, b) = must_carry_sum(10, 99, 10, 99)
# variables = [c,d]
# (c, d) = must_borrow_diff(100, 999, 10, 99)
# correct=[x, y, z, p, a, b, c, d]
# incorrect=[(x+y+1),(x+y-1),(x+y+10), (x+y-10), z]
# incorrect_length = 3
# incorrect_value > 0
# question = number
# """
# print(parse_pseudo_code(pseudo_code))


# pseudo_code = """
# variables = [x, y]
# variables = [z]
# x = range(1,100,3)
# y = [1,2]
# z = rev(x,y)
# correct = [x, y, z]
# """
# result = parse_pseudo_code(pseudo_code)
# print(result)

# pseudo_code = """
# variables = [x, y, z]
# x = range(1,10)
# y,z = range(1, x)
# a, b = [2, 3, 4, 5]
# unique_options = True
# answer = number
# order = ascending
# options = [1, 2, 3, 4, 5]
# options_length = 5
# carry_over = True
# question = number
# incorrect_length = 4
# borrow = [0, 1, 0]
# start_point = [x/y,2/3]
# end_point = [1/x,1/y,1]
# """
# result = parse_pseudo_code(pseudo_code)
# print(result)

# pseudo_code = """
# variables = [x, y, z]
# x = 2
# y = 3
# z = 4
# correct_type = fraction
# correct = [2(x+1)(y+1)(x+y)(x-y)/(3)(3x)(xy)(z), (x + (2x + (x+y)))/((x+1)(x-1)), (x(x+x))/(x(x(x+x)-x)) ]
# incorrect_type = decimal
# incorrect = [x/y, y/z, x/z]
# """
# result = parse_pseudo_code(pseudo_code)
# print(result)

# # Test the modified pseudo-code parsing
# pseudo_code = """
# variables = [x, y, z]
# x = 3
# y = range(1,x)
# z = range(y, 2x)
# a, b = [2, 3,4,5]
# correct_type = fraction
# correct = [(x+y)/(y+z), (y+z)/x]
# incorrect_type = decimal
# incorrect = [(x+1)/y, 3y/z, 4z/x]
# """
# result = parse_pseudo_code(pseudo_code)
# print(result)



# pseudo_code = """
# variables = [x, y, z]
# x = 2
# y = 3
# z = 4
# correct_type = fraction
# correct = [(x+y)/(y+z), (y+z)x]
# incorrect_type = decimal
# incorrect = [x/y, y/z, x/z]
# """
# result = parse_pseudo_code(pseudo_code)
# print(result)


# pseudo_code = """
# variables = [x]
# x = range(0,100,3)
# """
# result = parse_pseudo_code(pseudo_code)
# print(result)

