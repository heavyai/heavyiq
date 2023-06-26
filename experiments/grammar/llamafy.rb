#!/usr/bin/env ruby
#
# Copyright 2023 HEAVY.AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Transform the BNF Grammar for ISO/IEC 9075-2:2003 to be compatible w/ llama.cpp.
#
# Usage: ./llamafy.rb < sql-2003-2.bnf > sql-2003-2.gbnf
# Also works with sql-92.bnf.

content = ARGF.read

# Comment header lines prior to first comment
content.sub!(%r{\A.*?(?=^--)}m) { $&.gsub /^\S/, '--\0' }

# Remove tabs.
content.gsub! /\t/, '    '

# Comment comment lines.
content.gsub! /^--/, '#--'

# Comment lines betweep --p and --/p
# and for SQL-92: lines betweep --small and --/small
content.gsub! %r{(^#--(p|small)\n)(.*?)(^#--/\2\n)}m do
  tag = $2
  comment = $3.gsub(%r{^(?!#).*$}, '#\&')
  "#--#{tag}\n#{comment}#--/#{tag}\n"
end

angle_rule_regex = %r{(?<=\n\n)<.*?(?=\n\n)}m
# Count number of rules
nrules = content.scan(angle_rule_regex).size
STDERR.puts "nrules=#{nrules}"

# Separate adjacent names
content.gsub!(angle_rule_regex) { $&.gsub '><', '> <' }

# Remove angle brackets from names.
# Replace non-alphanumeric characters (spaces, dashes, slashes, colons) w/ dashes.
name_class = '[-/: \w]+'
content.gsub!(angle_rule_regex) do |rule|
  rule.gsub(Regexp.new "<(#{name_class})>") { $1.gsub(/\W/, '-') }
end

# Parenthesize multi-line definitions
rule_regex = %r{(?<=\n\n)\w.*?(?=\n\n)}m
content.gsub!(rule_regex) { $&.sub(%r{(.*?\n)(.*)}m, "\\1(\\2\n)") }

# Quote single-character definitions
# and SQL-92: <> >= <= || ..
content.gsub!(Regexp.new '^(%s ::= )(\S\W?)(?=\n\n)' % name_class, 'm') { "#$1#{$2.inspect}" }

# Quote trigraphs
content.gsub!(Regexp.new '^(%s?-trigraph ::= )(\S+)(?=\n\n)' % name_class, 'm') { "#$1#{$2.inspect}" }

# Quote tokens in Ada-qualified-type-specification
content.gsub! 'Interfaces.SQL', '"\&"'
content.gsub! /\bSQL_STANDARD\.\w+/, '"\&"' # SQL-92

content.gsub! '...omitted...', '"\&"' # SQL-92

# Replace {} w/ ()
content.gsub! '{', '('
content.gsub! '}', ')'

# Use * and + instead of ...
content.gsub!(Regexp.new '\[ (%s)\s*\.\.\. \]' % name_class) { "#$1*" }
content.gsub!(Regexp.new '\[ (\([^\[()\]]+\))\s*\.\.\. \]') { "#$1*" }
content.gsub!(Regexp.new '(%s)\s*\.\.\.' % name_class) { "#$1+" }
content.gsub!(Regexp.new '(\([^()]+\))\s*\.\.\.') { "#$1+" }
# Special case for binary-string-literal
content.gsub!(Regexp.new '\[ (\(.+?\))\s*\.\.\. \]') { "#$1*" }

# Replace [] w/ ()?
content.gsub!(Regexp.new '\[([^\[\]]+)\]') { "(#$1)?" }
content.gsub!(Regexp.new '\[([^\[\]]+)\]') { "(#$1)?" }
content.gsub!(Regexp.new '\[([^\[\]]+)\]') { "(#$1)?" }

# Words in ALLCAPS are assumed to be string values, not names
# Exceptions: hexit and simple-Latin-lower-case-letter have single lower case letters,
# SQL-condition has a ","
# Fortran-type-specification has an "="
content.gsub!(rule_regex) { $&.gsub(/(?<=\s)([-A-Z0-9_]+|[a-z,=])(?=\s|$)/m, '"\1"') }

# Fix "!! See the Syntax Rules"
syntax_rules = {
  'space' => '[ \t\n]+',
  'identifier-start' => '[a-zA-Z_]',
  'identifier-extend' => '[a-zA-Z0-9_]',
  'Unicode-escape-character' => '"TODO"',
  'nondoublequote-character' => '[^\"]', # ?
  'newline' => '"\n"',
  'nonquote-character' => "[^']",
  'non-escaped-character' => 'SQL-language-identifier-part', # TODO
  'escaped-character' => 'SQL-language-identifier-part', # TODO
  'preparable-implementation-defined-statement' => '"TODO"',
  'preparable-SQL-implementation-defined-statement' => '"TODO"', # SQL-92
  'SQLSTATE-class-value' => '"TODO"',
  'SQLSTATE-subclass-value' => '"TODO"',
  'host-label-identifier' => '"TODO"',
  'host-PL-I-label-variable' => '"TODO"',
  'embedded-SQL-Ada-program' => '"TODO"',
  'Ada-host-identifier' => '"TODO"',
  'embedded-SQL-C-program' => '"TODO"',
  'C-host-identifier' => '"TODO"',
  'embedded-SQL-COBOL-program' => '"TODO"',
  'embedded-SQL-Cobol-program' => '"TODO"', # SQL-92
  'COBOL-host-identifier' => '"TODO"',
  'Cobol-host-identifier' => '"TODO"', # SQL-92
  'embedded-SQL-Fortran-program' => '"TODO"',
  'Fortran-host-identifier' => '"TODO"',
  'embedded-SQL-MUMPS-program' => '"TODO"',
  'MUMPS-host-identifier' => '"TODO"',
  'embedded-SQL-Pascal-program' => '"TODO"',
  'Pascal-host-identifier' => '"TODO"',
  'embedded-SQL-PL-I-program' => '"TODO"',
  'PL-I-host-identifier' => '"TODO"',
  'direct-implementation-defined-statement' => '"TODO"'
}
content.gsub!(Regexp.new '^(%s)\s+::=.*' % syntax_rules.keys.join('|'), 'm') { "#$1 ::= #{syntax_rules[$1]}" }

# Define root as required by parser
content += "\nroot ::= direct-SQL-statement\n"

puts content
