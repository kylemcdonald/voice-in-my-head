import csv
from helpers import ThreadedJob, log


# combine consecutive "speak" rows into a single row
def preprocess_script(reader, language):
    rows = []
    prev_row = None
    for row in reader:
        # copy the input from the specified language if it's not empty
        if language != "en" and row[language] != "":
            row["input"] = row[language]
        if row["function"] == "speak":
            if prev_row is None:
                prev_row = row
            else:
                prev_row["input"] += f" {row['input']}"
        else:
            if prev_row is not None:
                rows.append(prev_row)
                prev_row = None
            rows.append(row)
    return rows


class ScriptReader:
    def __init__(self, script_fn, language):
        with open(script_fn) as csvfile:
            self.reader = csv.DictReader(csvfile)
            self.rows = preprocess_script(self.reader, language)
            self.index = 0
            self.memory = {}

    def get_memory(self, variable_name):
        # if we are looking for a variable that is a ThreadedJob, wait for it to finish
        if isinstance(self.memory[variable_name], ThreadedJob):
            log(f"waiting on {variable_name} to resolve")
            self.memory[variable_name] = self.memory[variable_name].wait()
            log(f"{variable_name} resolved")
        return self.memory[variable_name]

    def run_row(self, row):
        raw_function, raw_input, output = row["function"], row["input"], row["output"]

        input = raw_input
        is_async = raw_function.startswith("async ")
        function = raw_function[6:] if is_async else raw_function

        if hasattr(self, function):
            # first, if input is a number, convert it to a number
            try:
                input = float(input)
            except ValueError:
                pass
            # second, look up the input in memory in case it's a variable name
            if input in self.memory:
                input = self.get_memory(input)
            # third, replace any variables in the input with their values
            if input and isinstance(input, str):
                for key in self.memory:
                    if f"{{{key}}}" not in input:
                        continue
                    input = input.replace(f"{{{key}}}", self.get_memory(key))

            func = getattr(self, function)
            if input:
                log(f"{raw_function}({raw_input})")
                result = ThreadedJob(func, input) if is_async else func(input)
            else:
                log(f"{raw_function}()")
                result = ThreadedJob(func) if is_async else func()
            if output:
                print(f"{output} ← {result}")
                self.memory[output] = result
        else:
            print(function)
