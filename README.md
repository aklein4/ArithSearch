# ArithSearch

### Current Larget Polynomial Solved:
16c^2 + 16c + 16a^2c^2 + 32abc^2 + 16b^2c^2 + 4 + 8a^2c + 16abc + 8b^2c + 4a^4c^2 + 16a^3bc^2 + 24a^2b^2c^2 + 16ab^3c^2 + 4b^4c^2
(Cost: 54 combined + and *)

## circuit.py
A simple library for creating, running, and visualizing arithmetic circuits as trees.

## brute_force.py
A semi-naive method of brute-forcing polynomial circuits (made obsolete by smart_force.py, but is still easier to understand). Modify main() for use.

## smart_force.py
A much more complex version of brute-force search, running many orders of magnitude faster. Saves output to .csv file that can be read by read_lib.py. Modify constraint constant for use.

## read_lib.py
Reads the output files generated by smart_force.py to print resulting polynomials and computation trees to the terminal. To use, add filename as argument (ex. 'python3 read_lib.py example.csv'). Prompted inputs represent the index of the desired polynomial (from first column of input file).

### See commments for details.
