import nbformat
import os

def count_indentation_spaces(line):
    count = 0
    for char in line:
        if char == ' ':
            count += 1
        else:
            break
    return count

def is_import_line(line):
    return line.startswith('import ') or line.startswith('from ')

def is_function_definition(line):
    return line.startswith('def ')

def save_funcs(notebook_path, output_dir):
    # Read the notebook file
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)

    # Extract code cells
    code_cells = [cell for cell in notebook.cells if cell['cell_type'] == 'code']

    # Write code lines to a .py file
    script_path = os.path.join(output_dir, os.path.splitext(os.path.basename(notebook_path))[0] + '_funcs.py')
    with open(script_path, 'w') as f:
        for cell in code_cells:

            lines = cell['source'].splitlines()
            inside_function = False
            function_indentation = 0

            for line in lines:
                if is_import_line(line):          # Capture imports
                    f.write(line)
                    f.write('\n')
                if is_function_definition(line):  # Capture function definition
                    inside_function = True
                    function_indentation = count_indentation_spaces(line)
                if inside_function:               # Capture function contents
                    if line.strip() and count_indentation_spaces(line) <= function_indentation and not is_function_definition(line):
                        inside_function = False
                    else:
                      f.write(line)
                      f.write('\n')
            f.write('\n')