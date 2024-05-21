import os
import re
import sys

# Define directories
db_dir = 'database'
new_dir = 'database_modified'

# If new directory does not exist, create it
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

subdirs = [] 
if len(sys.argv) > 1:
  subdirs.append(sys.argv[1])
else:
    subdirs = os.listdir(db_dir)

# Iterate over all subdirectories in the main directory
for subdir in subdirs:
    path = os.path.join(db_dir, subdir)
    
    # Ensure that the path is indeed a directory
    if os.path.isdir(path):
        # Get .sql files
        for file in os.listdir(path):
            if file.endswith('.sql'):
                with open(os.path.join(path, file), 'r') as f:
                    lines = f.readlines()

                # Replace 'PRIMARY KEY' and 'VARCHAR' in lines
                lines = [line for line in lines if 'FOREIGN KEY' not in line and 'PRAGMA' not in line and 'UNIQUE (' not in line and not line.strip().startswith('PRIMARY KEY') and 'primary key (' not in line and 'primary key(' not in line and 'foreign key (' not in line and 'foreign key(' not in line]
                lines = ["""DROP DATABASE IF EXISTS {db_name};\n""".format(db_name=subdir), """CREATE DATABASE {db_name};\n""".format(db_name=subdir), """ALTER SESSION SET CURRENT_DATABASE = '{db_name}';\n""".format(db_name=subdir)] + lines

                #lines = [re.sub(r'\bPRIMARY KEY\b', '', line) for line in lines]
                lines = [re.sub(r'\bREAL', 'FLOAT', line) for line in lines]
                lines = [re.sub(r'\breal', 'FLOAT', line) for line in lines]
                lines = [re.sub(r'\bVARCHAR\b(\(\d+\))?', 'TEXT', line) for line in lines]
                lines = [re.sub(r'\bvarchar\b(\(\d+\))?', 'TEXT', line) for line in lines]
                lines = [re.sub(r'\bDATETIME', 'TIMESTAMP(0)', line) for line in lines]
                lines = [re.sub(r'\bdatetime', 'TIMESTAMP(0)', line) for line in lines]
                lines = [re.sub(r'DECIMAL \(19', 'DECIMAL (18', line) for line in lines]
                lines = [re.sub(r'DECIMAL\(19', 'DECIMAL (18', line) for line in lines]
                lines = [re.sub(r'\bPRIMARY KEY', '', line) for line in lines]
                lines = [re.sub(r'\bprimary key', '', line) for line in lines]
                lines = [re.sub(r'`', '', line) for line in lines]

                modified_lines = []
                for line in lines:
                    if 'VALUES (' in line:
                      values = line.split('VALUES (')[1][:-3].split(',')
                      new_values = []
                      for value in values:
                        new_value = value
                        if len(new_value) >= 2:
                          if new_value[0] == '"':
                             new_value[0] = "'"
                          if new_value[-1] == '"':
                             new_value[-1] = "'"
                          new_value = new_value[0] + new_value[1:-1].replace("'", "''") + new_value[-1]
                        new_values.append(new_value)
                      modified_line = line.split('VALUES (')[0] + 'VALUES (' + ','.join(new_values) + ');\n'
                      modified_lines.append(modified_line)
                    else:
                        modified_lines.append(line)

                data = "".join(modified_lines)

                # Find all CREATE TABLE statements and remove trailing commas
                data = re.sub(r'(CREATE TABLE .*?)\,\s*\)', r'\1)', data, flags=re.DOTALL)
                data = re.sub(r'(create table .*?)\,\s*\)', r'\1)', data, flags=re.DOTALL)

                # Create the new .sql file
                with open(os.path.join(new_dir, f'{subdir}.sql'), 'w') as f:
                    f.write(data)
