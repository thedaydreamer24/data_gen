import json
import requests
from datetime import datetime

from app import db
from app.managers import StreakManager, ContentManager
from app.models import (
    QuestionTemplate,
    Variant,
    Lesson,
)
from app.auth.constants import ErrorMessage
from app.exceptions.api_exceptions import HTTPUnprocessableEntity
from app.utils.parser import *
from app.utils.constants import (
    QuestionType,
    level_to_target_questions,
    game_format_to_speed,
    speed_to_level_change,
    speed_to_skip_enable,
)

# from app.managers import QuestionManager
# variant_ids = ['8733b724-2495-11ef-ac15-3ed868218b01']
# a = QuestionManager.create_json_structure(variant_ids)

class QuestionManager:
    
    @classmethod
    def generate_options(cls, context):
        options_created = False
        attributes = {}
        if "question" in context and context.get("question") == "number_line":
            start_point = int(context.get("start_point")[0])
            end_point = int(context.get("end_point")[0])
            divisions = end_point - start_point - 1
            interval_length = int(context.get("interval_length"))
            sub_divisions = int(context.get("n"))-1
            correct = context.get("correct")
            if not context.get("incorrect"):
                return {
                    "correct": correct,
                    "current_value": [str(c) for c in correct],
                    "options": [],
                    "question_type": QuestionType.FILL_IN_THE_BLANK,
                    "type_count": 1,
                }
            incorrect = context.get("incorrect")
            all_options = correct + incorrect
            all_options.sort()
            current_value = [str(c) for c in all_options]
            #index of correct answer in all_options
            correct_index = all_options.index(correct[0])
            # create all options as A,B,C,D
            all_options = [chr(65 + i) for i in range(len(all_options))]
            #randomly shuffle the options
            random.shuffle(all_options)
            options = []
            for i in range(len(all_options)):
                option = {
                    "option_id": chr(97 + i),
                    "type": "string",
                    "value": str(all_options[i]),
                    "type_count": "1",
                }
                options.append(option)
            # get the option_id of the chr(65+correct_index) value in options
            correct_answer = []
            for i in range(len(options)):
                if options[i].get("value") == chr(65 + correct_index):
                    correct_answer.append(options[i].get("option_id"))

            return {
                "correct": correct_answer,
                "options": options,
                "question_type": QuestionType.SINGLE_CHOICE,
                "current_value": current_value,
                "type_count": 1,
            }

        elif "start_point" in context and "end_point" in context and "interval_length" in context:
            # check interval_length is Fraction instance
            if isinstance(context.get("interval_length"), Fraction):
                start_point = Fraction(context.get("start_point")[0])
                end_point = Fraction(context.get("end_point")[0])
                interval_length = Fraction(context.get("interval_length"))

                # Generate the range using fractions
                current = start_point
                all_options = []
                while current <= end_point:
                    all_options.append(current)
                    current += interval_length
            else:
                start_point = int(context.get("start_point")[0])
                end_point = int(context.get("end_point")[0])
                interval_length = int(context.get("interval_length"))
                all_options = list(range(start_point, end_point + 1, interval_length))
            # convert the Fractions to strings if present
            all_options = [str(x) for x in all_options]
        elif "options" in context:
            if "options_length" in context:
                options_length = int(context.get("options_length"))
                all_options = context.get("options")
                # remove duplicates
                if "unique_options" in context and context.get("unique_options").lower() == "true":
                    all_options = list(set(all_options))
                if options_length < len(all_options):
                    all_options = all_options[:options_length]
                    all_options = [str(x) for x in all_options]
                elif options_length > len(all_options):
                    # repeat the options values to match the length
                    all_options = all_options * (options_length // len(all_options)) + all_options[:options_length % len(all_options)]
                    all_options = [str(x) for x in all_options]
                else:
                    all_options = [str(x) for x in all_options]
        elif "correct" in context and ("incorrect" in context or "incorrect_static" in context or "incorrect_dynamic" in context):
            correct = context.get("correct", [])

            incorrect_static = context.get("incorrect_static", [])  # P2
            incorrect_dynamic = context.get("incorrect_dynamic", [])  # P1
            incorrect = context.get("incorrect", [])

            incorrect_static = list(set(incorrect_static))
            incorrect_dynamic = list(set(incorrect_dynamic))
            incorrect = list(set(incorrect))  # Ensure incorrect is a unique list

            # Remove values which are equal to correct values
            incorrect_static = [x for x in incorrect_static if x not in correct]
            incorrect_dynamic = [x for x in incorrect_dynamic if x not in correct]
            incorrect = [x for x in incorrect if x not in correct]

            min_incorrect_value = context.get("min_incorrect_value", None)
            if min_incorrect_value is not None:
                min_incorrect_value = int(min_incorrect_value)
                incorrect_static = [x for x in incorrect_static if x > min_incorrect_value]
                incorrect_dynamic = [x for x in incorrect_dynamic if x > min_incorrect_value]
                incorrect = [x for x in incorrect if x > min_incorrect_value]

            incorrect_length = int(context.get("incorrect_length", 0))

            # Handle positive_incorrect and negative_incorrect
            positive_incorrect = context.get("positive_incorrect", 0)
            negative_incorrect = context.get("negative_incorrect", 0)

            if positive_incorrect > 0 or negative_incorrect > 0:
                if incorrect_dynamic or incorrect_static:
                    # If dynamic or static options are available
                    if incorrect_dynamic:
                        positive_incorrect_dynamic = [x for x in incorrect_dynamic if x > 0]
                        negative_incorrect_dynamic = [x for x in incorrect_dynamic if x < 0]
                    else:
                        positive_incorrect_dynamic = []
                        negative_incorrect_dynamic = []

                    if incorrect_static:
                        additional_positive_static = [x for x in incorrect_static if x > 0]
                        additional_negative_static = [x for x in incorrect_static if x < 0]
                    else:
                        additional_positive_static = []
                        additional_negative_static = []

                    # Ensure we have enough positive and negative incorrect values
                    positive_from_dynamic = positive_incorrect_dynamic[:positive_incorrect]
                    negative_from_dynamic = negative_incorrect_dynamic[:negative_incorrect]

                    # Calculate remaining slots for positive and negative incorrect options
                    remaining_positive = positive_incorrect - len(positive_from_dynamic)
                    remaining_negative = negative_incorrect - len(negative_from_dynamic)

                    if remaining_positive > 0:
                        positive_from_static = random.sample(additional_positive_static, min(remaining_positive, len(additional_positive_static)))
                        positive_from_dynamic.extend(positive_from_static)
                    
                    if remaining_negative > 0:
                        negative_from_static = random.sample(additional_negative_static, min(remaining_negative, len(additional_negative_static)))
                        negative_from_dynamic.extend(negative_from_static)

                    # Combine positive and negative options
                    all_incorrect = positive_from_dynamic + negative_from_dynamic

                    # Fill the rest of incorrect options if there is still space
                    if len(all_incorrect) < incorrect_length:
                        additional_needed = incorrect_length - len(all_incorrect)
                        additional_incorrect = [x for x in incorrect_dynamic if x not in all_incorrect]
                        additional_incorrect += [x for x in incorrect_static if x not in all_incorrect]
                        all_incorrect.extend(random.sample(additional_incorrect, min(additional_needed, len(additional_incorrect))))
                
                else:
                    # If no dynamic or static options are available, use only incorrect list
                    positive_incorrect_list = [x for x in incorrect if x > 0]
                    negative_incorrect_list = [x for x in incorrect if x < 0]

                    positive_from_list = positive_incorrect_list[:positive_incorrect]
                    negative_from_list = negative_incorrect_list[:negative_incorrect]

                    # Calculate remaining slots for positive and negative incorrect options
                    remaining_positive = positive_incorrect - len(positive_from_list)
                    remaining_negative = negative_incorrect - len(negative_from_list)

                    if remaining_positive > 0:
                        additional_positive = [x for x in incorrect if x > 0 and x not in positive_from_list]
                        positive_from_list.extend(random.sample(additional_positive, min(remaining_positive, len(additional_positive))))
                    
                    if remaining_negative > 0:
                        additional_negative = [x for x in incorrect if x < 0 and x not in negative_from_list]
                        negative_from_list.extend(random.sample(additional_negative, min(remaining_negative, len(additional_negative))))

                    # Combine positive and negative options
                    all_incorrect = positive_from_list + negative_from_list

                    # Fill the rest of incorrect options if there is still space
                    if len(all_incorrect) < incorrect_length:
                        additional_needed = incorrect_length - len(all_incorrect)
                        additional_incorrect = [x for x in incorrect if x not in all_incorrect]
                        all_incorrect.extend(random.sample(additional_incorrect, min(additional_needed, len(additional_incorrect))))
                
            else:
                if incorrect:
                    # Use only the incorrect values if they are provided
                    all_incorrect = random.sample(incorrect, min(incorrect_length, len(incorrect)))
                else:
                    # Prepare the incorrect options list from incorrect_static and incorrect_dynamic
                    if len(incorrect_dynamic) < incorrect_length:
                        # Calculate how many more options are needed from incorrect_static
                        needed_from_static = incorrect_length - len(incorrect_dynamic)
                        # Select random incorrect_static options to fill the gap
                        random_static = random.sample(incorrect_static, min(needed_from_static, len(incorrect_static)))
                        all_incorrect = incorrect_dynamic + random_static
                    else:
                        all_incorrect = incorrect_dynamic[:incorrect_length]

                # If the length of all_incorrect is still less than incorrect_length, extend it with additional values from incorrect
                if len(all_incorrect) < incorrect_length:
                    additional_needed = incorrect_length - len(all_incorrect)
                    additional_incorrect = [x for x in incorrect if x not in all_incorrect]
                    all_incorrect.extend(random.sample(additional_incorrect, min(additional_needed, len(additional_incorrect))))
            all_incorrect = all_incorrect[:incorrect_length]
            all_options = all_incorrect + correct
            all_options = [str(x) for x in all_options]

        elif "question" in context and context.get("question").startswith("input_clock"):
            # "attributes": {
            #         "sub_type": "clock_1",
            #         "meridiem": "am", (am/pm),
            #         "hour_hand_error_precision": [60000,60000] ([0,0] by default),
            #         "minute_hand_error_precision": [60000,60000] ([0,0] by default),
            #       }
            print("**********", flush=True)
            possible_clock_types = ["clock_1", "clock_2", "clock_3", "clock_4"]
            clock_types_given = context.get("question").replace("input_clock_", "").split("_")
            clocks = ["clock_"+str(x) for x in clock_types_given]
            selected_clock = random.choice(clocks)
            sub_type = selected_clock
            meridiem = random.choice(["am", "pm"])
            hour_hand_error_precision = [0, 0]
            minute_hand_error_precision = [0, 0]
            degree_to_milliseconds = 60000
            if "error_margin" in context:
                error_margin = context.get("error_margin")
                hour_hand_error_precision = [int(error_margin[0]) * degree_to_milliseconds, int(error_margin[1]) * degree_to_milliseconds]
                minute_hand_error_precision = [int(error_margin[2]) * degree_to_milliseconds, int(error_margin[3]) * degree_to_milliseconds]
            attributes["sub_type"] = sub_type
            attributes["meridiem"] = meridiem
            attributes["hour_hand_error_precision"] = hour_hand_error_precision
            attributes["minute_hand_error_precision"] = minute_hand_error_precision
            option_type = "string"
            question_value = str(context.get("question_value", ""))
            evaluated_question_value = eval(question_value)
            print("EVALUATED QUESTION VALUE", evaluated_question_value, flush=True)
            if isinstance(evaluated_question_value, list):
                question_value = evaluated_question_value[0]
            elif isinstance(evaluated_question_value, int):
                question_value = evaluated_question_value
            question_value = str(question_value*60000)
            all_options = [question_value]
            shape_count = 1
            options_created = True
            options = []
            for i in range(len(all_options)):
                option = {
                    "option_id": chr(97 + i),
                    "type": option_type,
                    "value": str(all_options[i]),
                    "type_count": shape_count,
                    "attributes": attributes
                }
                options.append(option)
            response = {
            "correct": ["a"],
            "options": options,
            "question_type": QuestionType.SINGLE_CHOICE,
            "type_count": 1,
            "attributes": attributes
            }
            return response
        elif "answer" in context and context.get("answer") == "boolean":
            correct = context.get("correct")[0]
            options = [
                {"option_id": "a", "type": "boolean", "value": "True", "type_count": 1},
                {"option_id": "b", "type": "boolean", "value": "False", "type_count": 1}
            ]
            
            correct_answer = ["a"] if correct else ["b"]
            context["options_type"] = "boolean"
            
            return {
                "correct": correct_answer,
                "options": options,
                "question_type": QuestionType.SINGLE_CHOICE,
                "type_count": 1,
                "value": "True" if correct else "False"
            }
        else:
            correct = context.get("correct")
            correct = [str(x) for x in correct]
            return {
                "correct": correct,
                "options": [],
                "question_type": QuestionType.FILL_IN_THE_BLANK,
                "type_count": 1,
            }
        # correct = context.get("correct")
        # correct = [str(x) for x in correct]
        if "start_point" in context and "end_point" in context and "interval_length" in context:
            pass
        else:
            random.shuffle(all_options)
        options = []
        option_type = "string"
        shape_count = 1
        if "answer" in context and context.get("answer").startswith("shape"):
            correct_value = context.get("correct")
            shapes = ["circle", "rectangle"]
            no_of_partitions = 2
            if isinstance(correct_value, str):
                no_of_partitions = int(str(correct_value).split("/")[1])
                correct_value_str = str(correct_value)
            elif isinstance(correct_value, list):
                no_of_partitions = int(str(correct_value[0]).split("/")[1])
                correct_value_str = str(correct_value[0])
            if no_of_partitions != 2:
                shapes.append("polygon")
            else:
                print("No of partitions", no_of_partitions, flush=True)
            shape = str(context.get("answer"))
            shape_count = int(shape.split("_")[1])
            if "shape_merge" in context:
                no_of_shape_merges = int(context.get("shape_merge").strip())
                incorrect_shapes = ["circle", random.choice(["rectangle", "polygon"])]
                for i in range(len(all_options)):
                    merge = None
                    is_correct_value_str = str(all_options[i]) == correct_value_str
                    if is_correct_value_str:
                        option_type = random.choice(shapes)
                        current_shape = option_type
                        shapes.remove(option_type)
                        if not shapes:
                            shapes = ["circle", "rectangle"]
                    else:
                        current_shape = incorrect_shapes[0]
                    if incorrect_shapes[0] == "circle" and not is_correct_value_str:
                        merge = 2
                        no_of_shape_merges -= 1
                        if no_of_shape_merges > 0:
                            incorrect_shapes[0] = "circle"
                        else:
                            incorrect_shapes[0] = random.choice(["rectangle", "polygon"])
                    incorrect_shapes.append(random.choice(["rectangle", "polygon"]))
                    option = {
                        "option_id": chr(97 + i),
                        "type": current_shape,
                        "value": str(all_options[i]),
                        "type_count": shape_count
                    }
                    if merge:
                        option["merge"] = merge
                    options.append(option)
            options_created = True

        if "answer" in context and context.get("answer") == "fraction":
            option_type = "fraction"


        if "answer" in context and context.get("answer") == "number":
            print("ALL OPTIONS", all_options, flush=True)
            all_options = [str(int(eval(str(x).replace('÷', '/').replace('×', '*')))) for x in all_options]
            print("ALL OPTIONS2", all_options, flush=True)
        
            
        if "answer" in context and context.get("answer") == "time":
            new_all_options = []
            for i in range(len(all_options)):
                val = all_options[i]
                hour = str(int(int(val)/60))
                minute = str(int(int(val)%60))
                if hour == "0":
                    hour = "12"
                if len(minute) == 1:
                    minute = "0" + minute
                new_all_options.append(hour + ":" + minute)
            all_options = new_all_options
            option_type = "string"

        if "answer" in context and context.get("answer").strip().startswith("clock"):
            print("ANSWER_CLOCK", flush=True)
            clock_types_given = context.get("answer").replace("clock_", "").split("_")
            clocks = ["clock_"+str(x) for x in clock_types_given]
            selected_clock = random.choice(clocks)
            sub_type = selected_clock
            attributes["sub_type"] = sub_type
            option_type = "clock"
            all_options = [str(int(x)*60000) for x in all_options]
            context["options_type"] = "no_eval"

        if not options_created:
            for i in range(len(all_options)):
                option = {
                    "option_id": chr(97 + i),
                    "type": option_type,
                    "value": str(all_options[i]),
                    "type_count": shape_count
                }
                options.append(option)

        show_labels = True
        if "show_labels" in context and context.get("show_labels").lower() == "false":
            show_labels = False
        if "start_point" in context and "end_point" in context and "interval_length" in context:
            for option in options:
                possible_values = ["0", "1", "2", "3"] + [all_options[-1]]
                if option.get("value") in possible_values:
                    if "hide_integers" in context and context.get("hide_integers", "").lower() == "true" and option.get("value") != "0":
                        option["show_label"] = False
                    else:
                        option["show_label"] = True
                else:
                    option["show_label"] = show_labels
                possible_values = [all_options[0], all_options[-1]]
                print("POSSIBLE VALUES", possible_values, flush=True)
                if "show_endpoints" in context and context.get("show_endpoints", "").lower() == "true" and option.get("value") in possible_values:
                    option["show_label"] = True

        if "order" in context:
            order = context.get("order") # any, ascending, descending
            if order == "any":
                order = random.choice(["ascending", "descending"])
            ascending = order == "ascending"
            if context.get("options_type", "") == "no_eval":
                def eval_with_abs(expression):
                    def abs_replacer(match):
                        inner_expr = match.group(1)
                        return str(abs(eval(inner_expr.replace('÷', '/').replace('×', '*'))))

                    pattern = re.compile(r'\|([^|]+)\|')
                    while pattern.search(expression):
                        expression = pattern.sub(abs_replacer, expression)

                    return eval(expression.replace('÷', '/').replace('×', '*').replace(':', '/'))
                evaluated_options = []
                for option in options:
                    value = option.get("value").replace('÷', '/').replace('×', '*').replace(':', '/')
                    try:
                        evaluated_value = eval(value)
                    except SyntaxError:
                        evaluated_value = eval_with_abs(value)
                    evaluated_options.append(evaluated_value)

                # evaluated_options = [eval(option.get("value").replace('÷', '/'),  {"__builtins__": None, "int": int, "Fraction": Fraction}, context) for option in options]
                # print("EVALUATED OPTIONS", evaluated_options, flush=True)
                evaluated_options.sort(reverse=not ascending)
                correct_answer = [option.get("option_id") for option in options]
                while correct_answer == ["a", "b", "c", "d"]:
                    random.shuffle(correct_answer)
                    
                for i in range(len(all_options)):
                    value = evaluated_options[i]
                    option_id = None
                    for option in options:
                        option_value = option.get("value").replace('÷', '/').replace('×', '*').replace(':', '/')
                        try:
                            option_eval = eval(option_value)
                        except SyntaxError:
                            option_eval = eval_with_abs(option_value)
                        if option_eval == value:
                            option_id = option.get("option_id")
                            break
                    correct_answer[i] = option_id
                for i in range(len(options)):
                    # add space before and after '+', '-', 'x', '÷'
                    value = options[i].get("value").replace(" ", "")
                    value = value.replace("+", " + ").replace("-", " - ").replace("×", " × ").replace("÷", " ÷ ").replace(":", " : ")
                    options[i]["value"] = value
            else:
                all_options.sort(reverse=not ascending)
                all_options = [str(x) for x in all_options]
                correct_answer = []
                for i in range(len(all_options)):
                    value = all_options[i]
                    # find the option_id of the value in the options
                    option_id = None
                    for option in options:
                        if option.get("value") == value:
                            option_id = option.get("option_id")
                            break
                    correct_answer.append(option_id)
                    while correct_answer == ["a", "b", "c", "d"]:
                        random.shuffle(correct_answer)
            response = {
                "correct": correct_answer,
                "options": options,
                "question_type": QuestionType.ORDER_SEQUENCE,
                "order_type": order,
                "type_count": 1,
            }
            if attributes:
                response["attributes"] = attributes
            return response
        if "compare" in context:
            compare = context.get("compare")
            correct_answer = []
            print("ALL OPTIONS", all_options, flush=True)
            if compare == "greatest_or_least":
                #randomly choose greatest or least
                compare = random.choice(["greatest", "least"])
            if compare == "greatest":
                #find the option_id of the largest value in the options
                compare_values = [eval(option.replace('÷', '/').replace('×', '*')) for option in all_options]
                value_index = 0
                max_value = compare_values[0]
                for i in range(len(all_options)):
                    value = compare_values[i]
                    max_value = max(max_value, value)
                    if max_value == value:
                        value_index = i
                largest = all_options[value_index]
                print("MAX", largest, flush=True)
                for option in options:
                    if option.get("value") == largest:
                        print("OPTION", option.get("value"), flush=True)
                        correct_answer.append(option.get("option_id"))
                        break
            elif compare == "least":
                #find the option_id of the smallest value in the options
                compare_values = [eval(option.replace('÷', '/').replace('×', '*')) for option in all_options]
                value_index = 0
                min_value = compare_values[0]
                for i in range(len(all_options)):
                    value = compare_values[i]
                    min_value = min(min_value, value)
                    if min_value == value:
                        value_index = i
                smallest = all_options[value_index]
                print("MIN", smallest, flush=True)
                for option in options:
                    if option.get("value") == smallest:
                        print("OPTION", option.get("value"), flush=True)
                        correct_answer.append(option.get("option_id"))
                        break
            response = {
                "correct": correct_answer,
                "options": options,
                "question_type": QuestionType.SINGLE_CHOICE,
                "compare_type": compare,
                "type_count": 1,
            }
            if attributes:
                response["attributes"] = attributes
            return response

        # find the correct answers(list)
        correct = context.get("correct")
        correct = [str(x) for x in correct]
        if "correct_equivalent" in context and context.get("correct_equivalent", "").lower() == "true":
            new_correct = []
            for c in correct:
                # find the value in all_options which is equivalent to c
                for option in options:
                    # print("OPTION", option.get("value"), c, flush=True)
                    # print("types", type(option.get("value")), type(c), flush=True)
                    val1 = str(option.get("value"))
                    val2 = str(c)
                    if '/' not in val1:
                        val1 = val1 + "/1"
                    if '/' not in val2:
                        val2 = val2 + "/1"
                    if Fraction(int(val1.split('/')[0]), int(val1.split('/')[1])) \
                    == Fraction(int(val2.split('/')[0]), int(val2.split('/')[1])):
                        new_correct.append(str(option.get("value")))
                        break
            correct = new_correct
        if "answer" in context and context.get("answer") == "number":
            correct = [str(int(eval(x.replace('÷', '/').replace('×', '*')))) for x in correct]
        if "answer" in context and context.get("answer") == "time":
            new_correct = []
            for i in range(len(correct)):
                val = correct[i]
                hour = str(int(int(val)/60))
                minute = str(int(int(val)%60))
                if hour == "0":
                    hour = "12"
                if len(minute) == 1:
                    minute = "0" + minute
                new_correct.append(hour + ":" + minute)
            correct = new_correct
        elif "answer" in context and context.get("answer").strip().startswith("clock"):
            print("ANSWER_CLOCK2", flush=True)
            new_correct = []
            for val in correct:
                milliseconds = str(int(val) * 60000)
                new_correct.append(milliseconds)
            correct = new_correct
            print(correct)
            if attributes:
                for option in options:
                    option["attributes"] = attributes
        correct_answer = []
        for i in range(len(options)):
            if options[i].get("value") in correct:
                correct_answer.append(options[i].get("option_id"))
        question_type = QuestionType.SINGLE_CHOICE if len(correct_answer) == 1 else QuestionType.MULTIPLE_CHOICE
        if "incorrect_length" in context and context.get("incorrect_length") == "9":
            # replace one of the incorrect answer option with the correct answer
            correct_answer_ids = correct_answer
            incorrect_idx = 0
            correct_idx = 0
            for i in range(len(options)):
                if options[i].get("option_id") in correct_answer_ids:
                    correct_idx = i
                else: 
                    incorrect_idx = i
            options[incorrect_idx] = options[correct_idx]
        response = {
            "correct": correct_answer,
            "options": options,
            "question_type": question_type,
            "type_count": 1,
        }
        if attributes:
            response["attributes"] = attributes
        return response

    @classmethod
    def generate_questions(cls, num_questions, question_template):
        question_constraints = question_template.question_constraints
        answer_constraints = question_template.answer_constraints
        question_text = question_template.question_text.get("question_text")
        question_text = question_text.replace('{{', '{').replace('}}', '}')

        q_rules = question_constraints.get("q_rules")
        extra_rules = question_constraints.get("extra_rules")
        a_rules = answer_constraints.get("a_rules")
        # club the content in all the above rules line by line
        rules = f"{q_rules}\n{extra_rules}\n{a_rules}"
        #remove the empty lines
        rules = "\n".join([line for line in rules.split("\n") if line.strip()])
        # print("RULES", rules, flush=True)
        # context = parse_pseudo_code(rules)
        # print("RESULT", context, flush=True)

        # context = {'variables': '[z]', 'x': 4, 'y': 15, 'question': 'number', 'z': 514, 'correct': [19],
        # 'incorrect': [20, 18, 29, 9, 514, 24, 14, 515, 513], 'incorrect_length': '9', 'min_incorrect_value': '0'}

        questions = {}
        for i in range(1, num_questions + 1):
            context = parse_pseudo_code(rules)
            # if any of incorrect values are less than or equal to min_incorrect_value,
            # or greater than or equal to max_incorrect_value, regenerate the question
            while True:
                if "min_incorrect_value" in context:
                    min_incorrect_value = int(context.get("min_incorrect_value"))
                    if any([x <= min_incorrect_value for x in context.get("incorrect")]):
                        context = parse_pseudo_code(rules)
                        continue
                if "max_incorrect_value" in context:
                    max_incorrect_value = int(context.get("max_incorrect_value"))
                    if any([x >= max_incorrect_value for x in context.get("incorrect")]):
                        context = parse_pseudo_code(rules)
                        continue
                break

            text = question_text.format(**context)
            question_data = cls.generate_options(context)
            question_type = question_data.get("question_type")
            options = question_data.get("options")
            correct_answer = question_data.get("correct")
            question_value = str(context.get("question_value", ""))
            text_type = "string"
            shape_count = ""
            attributes = {}
            current_value = question_data.get("current_value", [])
            if "question" in context and context.get("question") == "number_line":
                start_point = int(context.get("start_point")[0])
                end_point = int(context.get("end_point")[0])
                divisions = end_point - start_point - 1
                sub_divisions = int(context.get("n"))-1
                text_type = "number_line"
                questions[str(i)] = {
                    "type": question_type,
                    "data": {
                        "text": text,
                        "type": text_type,
                        "value": current_value,
                        "options": options,
                        "correct_answer": correct_answer,
                        "start_point": start_point,
                        "end_point": end_point,
                        "divisions": divisions,
                        "sub_divisions": sub_divisions,
                        "type_count": 1,
                    }
                }
                continue
            elif "question" in context and context.get("question").startswith("shape"):
                shapes = ["circle", "rectangle"]
                no_of_partitions = 2
                if isinstance(question_value, str):
                    no_of_partitions = int(question_value.split("/")[1])
                elif isinstance(question_value, list):
                    no_of_partitions = int(question_value[0].split("/")[1])
                if no_of_partitions != 2:
                    shapes.append("polygon")
                else:
                    print("No of partitions", no_of_partitions, flush=True)
                # print("SHAPE", context.get("question"), flush=True)
                shape = str(context.get("question"))
                shape_count = int(shape.split("_")[1])
                text_type = random.choice(shapes)
            elif "question" in context and context.get("question") == "fraction":
                text_type = "fraction"
            elif "question" in context and context.get("question") == "time":
                hour = str(int(int(question_value)/60))
                min = str(int(int(question_value)%60))
                if hour == "0":
                    hour = "12"
                if len(min) == 1:
                    min = "0" + min
                text = question_text.replace("{x}", hour).replace("{y}", min)
                question_value = str(int(question_value)*60000)

            elif "question" in context and context.get("question").startswith("clock"):
                text_type = "clock"
                possible_clock_types = ["clock_1", "clock_2", "clock_3", "clock_4"]
                clock_types_given = context.get("question").replace("clock_", "").split("_")
                clocks = ["clock_"+str(x) for x in clock_types_given]
                selected_clock = random.choice(clocks)
                sub_type = selected_clock
                attributes["sub_type"] = sub_type
                evaluated_question_value = eval(question_value)
                # print("EVALUATED QUESTION VALUE", evaluated_question_value, flush=True)
                if isinstance(evaluated_question_value, list):
                    question_value = evaluated_question_value[0]
                elif isinstance(evaluated_question_value, int):
                    question_value = evaluated_question_value
                question_value = str(question_value*60000)
            elif "question" in context and context.get("question").startswith("input_clock"):
                # "attributes": {
                #         "sub_type": "clock_1",
                #         "meridiem": "am", (am/pm),
                #         "hour_hand_error_precision": [60000,60000] ([0,0] by default),
                #         "minute_hand_error_precision": [60000,60000] ([0,0] by default),
                #       }
                possible_clock_types = ["clock_1", "clock_2", "clock_3", "clock_4"]
                clock_types_given = context.get("question").replace("input_clock_", "").split("_")
                clocks = ["clock_"+str(x) for x in clock_types_given]
                selected_clock = random.choice(clocks)
                sub_type = selected_clock
                meridiem = random.choice(["am", "pm"])
                hour_hand_error_precision = [0, 0]
                minute_hand_error_precision = [0, 0]
                degree_to_milliseconds = 60000
                if "error_margin" in context:
                    error_margin = context.get("error_margin")
                    hour_hand_error_precision = [int(error_margin[0]) * degree_to_milliseconds, int(error_margin[1]) * degree_to_milliseconds]
                    minute_hand_error_precision = [int(error_margin[2]) * degree_to_milliseconds, int(error_margin[3]) * degree_to_milliseconds]
                attributes["sub_type"] = sub_type
                attributes["meridiem"] = meridiem
                attributes["hour_hand_error_precision"] = hour_hand_error_precision
                attributes["minute_hand_error_precision"] = minute_hand_error_precision
                text_type = "string"
                evaluated_question_value = eval(question_value)
                # print("EVALUATED QUESTION VALUE", evaluated_question_value, flush=True)
                if isinstance(evaluated_question_value, list):
                    question_value = evaluated_question_value[0]
                elif isinstance(evaluated_question_value, int):
                    question_value = evaluated_question_value
                hour = str(int(int(question_value)/60))
                minute = str(int(int(question_value)%60))
                if hour == "0":
                    hour = "12"
                if len(minute) == 1:
                    minute = "0" + minute
                if "time_delta" not in context:
                    text = question_text.replace("{x}", hour).replace("{y}", minute)
                question_value = str(question_value*60000)

                    
            if question_type == QuestionType.ORDER_SEQUENCE:
                # text = Arrange from <smallest/largest> to <largest/smallest>
                order_type = question_data.get("order_type")
                if order_type == "ascending":
                    text = text.replace("<smallest/largest>", "smallest").replace("<largest/smallest>", "largest")
                    text = text.replace("<increasing/decreasing>", "increasing").replace("<decreasing/increasing>", "increasing")
                else:
                    text = text.replace("<smallest/largest>", "largest").replace("<largest/smallest>", "smallest")
                    text = text.replace("<increasing/decreasing>", "decreasing").replace("<decreasing/increasing>", "decreasing")
                if "answer" in context and context.get("answer").startswith("clock_"):
                    am_pm = random.choice(["AM", "PM"])
                    text = text.replace("<AM/PM>", am_pm)
            if question_data.get("compare_type"):
                compare_type = question_data.get("compare_type")
                if compare_type == "greatest":
                    text = text.replace("<greatest/least>", "greatest")
                else:
                    text = text.replace("<greatest/least>", "least")
            questions[str(i)] = {
                "type": question_type,
                "data": {
                    "text": text,
                    "type": text_type,
                    "value": question_value,
                    "options": options,
                    "correct_answer": correct_answer,
                    "type_count": shape_count if shape_count else 1,
                }
            }
            if attributes:
                questions[str(i)]["data"]["attributes"] = attributes
        
        questions = cls.distribute_questions(questions)
        return questions
    
    @classmethod
    def distribute_questions(cls, questions):
        from collections import defaultdict
        import itertools
        
        # Group questions by their text content
        grouped_questions = defaultdict(list)
        for key, question in questions.items():
            grouped_questions[question['data']['text']].append(question)
        
        # Prepare for distribution
        distributed = []
        # Use round-robin to evenly distribute the questions
        for group in itertools.zip_longest(*grouped_questions.values()):
            distributed.extend([q for q in group if q is not None])

        # Re-index questions in the new order
        new_questions = {}
        for i, question in enumerate(distributed, start=1):
            new_questions[str(i)] = question
        
        return new_questions


    @classmethod
    def generate_levels(cls, speed, num_levels, question_templates):
        level_to_question_template = {}
        for question_template in question_templates:
            level = str(question_template.level)
            level = level.replace("LEVEL_", "")
            if level not in level_to_question_template:
                level_to_question_template[level] = []
            level_to_question_template[level].append(question_template)
        target_questions = level_to_target_questions.get(speed).get(num_levels)
        print("TARGET QUESTIONS", target_questions, flush=True)
        print("LEVEL TO QUESTION TEMPLATE", level_to_question_template, flush=True)
        levels = {}
        for i in range(1, num_levels + 1):
            question_template = level_to_question_template.get(str(i))[0]
            levels[f"level_{i}"] = cls.generate_questions(target_questions, question_template)
        return levels
    
    @classmethod
    def get_level_change_and_skip_enable(cls, variant_id_to_variant, variant_ids):
        # print("VARIANT IDS", variant_ids, flush=True)
        # print("VARIANT ID TO VARIANT", variant_id_to_variant, flush=True)
        level_change = []
        skip_enable = []
        for variant_id in variant_ids:
            variant = variant_id_to_variant.get(variant_id)
            if not variant:
                print("VARIANT NOT FOUND", variant_id, flush=True)
                # print(variant_id_to_variant, flush=True)
            game_format = variant.game_format
            speed = game_format_to_speed.get(game_format)
            level_change.append(speed_to_level_change.get(speed))
            skip_enable.append(speed_to_skip_enable.get(speed))
        return level_change, skip_enable


    @classmethod
    def create_json_structure(cls, variant_ids):
        variants = Variant.query.filter(Variant.id.in_(variant_ids)).all()
        # print("VARIANT IDS", variant_ids, flush=True)
        all_question_templates = QuestionTemplate.query.all()
        # query_result =  db.session.query(Variant, QuestionTemplate) \
        #     .join(QuestionTemplate, QuestionTemplate.variant_id == Variant.variant_id) \
        #         .filter(Variant.id.in_(variant_ids)).all()
        variant_id_to_variant = {}
        for variant in variants:
            variant_id_to_variant[str(variant.id)] = variant
        variant_id_to_question_templates = {}
        for variant in variants:
            variant_id_to_question_templates[str(variant.id)] = []
            for question_template in all_question_templates:
                if question_template.variant_id == str(variant.variant_id):
                    variant_id_to_question_templates[str(variant.id)].append(question_template)

        num_variants = len(variant_ids)
        num_levels = 3
        level_change, skip_enable = cls.get_level_change_and_skip_enable(
            variant_id_to_variant,
            variant_ids,
        )
        json_data = {
            "questions": {
                "level_change": level_change,
                "skip_enable": skip_enable,
                # "assets": {
                #     "image1": "url1",
                #     "image2": "url2"
                # }
                "assets": {}
            }
        }
        for i in range(num_variants):
            print("VARIANT", variant_ids[i], flush=True)
            variant = variant_id_to_variant.get(variant_ids[i])
            speed = game_format_to_speed.get(variant.game_format)
            # print(variant, flush=True)
            variant_internal_id = variant.variant_id
            # print("VARIANT INTERNAL ID", variant_internal_id, flush=True)
            variant_question_templates = variant_id_to_question_templates.get(variant_ids[i])
            num_levels = len(variant_question_templates)
            variant_key = f"variant_{variant_internal_id}"
            json_data["questions"][variant_key] = cls.generate_levels(speed, num_levels, variant_question_templates)

        # print("JSON DATA", json_data, flush=True)

        json_data = cls.remove_fractions_recursively(json_data)

        return json_data
    

    @classmethod
    def remove_fractions_recursively(cls,data):
        """
        Recursively converts all Fraction instances in a data structure to strings.
        
        Args:
            data (dict, list, or any): The data structure possibly containing Fraction instances.
            
        Returns:
            The modified data structure with all Fraction instances converted to strings.
        """
        if isinstance(data, dict):
            return {key: cls.remove_fractions_recursively(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [cls.remove_fractions_recursively(item) for item in data]
        elif isinstance(data, Fraction):
            return str(data)
        return data

# from app import db
# from app.managers import QuestionManager
# variant_ids = ['8715cf0c-2495-11ef-ac15-3ed868218b01']
# a = QuestionManager.create_json_structure(variant_ids)