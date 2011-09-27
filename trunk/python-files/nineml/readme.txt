Things that require testing/comments:

1.  A lexical parser for ODE equations/aliases and conditional expressions (in expression_parser.py) 
    and a set of classes that it is based on (in parser_objects.py).
    In general, the idea is to have the following features:
        - parsing the expressions given as strings from xml files
        - export in textual/Latex format
        - generation of equations/conditons for DAE Tools (ODEs, aliases, conditional expressions)
    The parser is quite generic (built with python ply package), and it accepts as an input the expression string. 
    The parser generates an abstract syntax tree and supports the following mathematical (+, -, *, /, **), 
    conditional (<, <=, >, >=, ==, !=) and logical operators (and, or) and basic mathematical functions 
    (sin, cos, tan, exp, sqrt, log10, log, ...). The AST can be evaluated by using two dictionaries: 
    Identifiers ('identifier-name' : value) and Functions ('function-name' : callable-object) or exported into another format 
    (string or Latex). The values and callable-objects have to support basic operators and functions as well 
    (to be able to evaluate the expression). This way the AST can be evaluated either by using simple numbers as values and 
    python math functions, or by using user defined objects that support math. operators and functions (to generate equations/logical 
    conditions for DAE Tools based on expression strings in NineML component specification). 
    There is some additional info in the source code files. There are already many tests, however it would be a good idea 
    to have it tested by more people.

2.  A lexical parser for units (in units_parser.py) and set of classes that it is based on (in parser_objects.py).
    In general, the idea is to have the following features:
        - parsing the units given as strings from xml files
        - exporting units in textual/Latex format
        - checking consistency of equations (ODEs, aliases, conditional expressions)
        - units conversion
        - setting the value in different units (value in meters to the quantity in kilometers for instance)
    The parser is based on the same class hierarchy as the expression parser and generates an abstract syntax tree 
    and supports the following mathematical operators: *, /, ** 
    The AST can be evaluated by providing a dictionary with base units: dictBaseUnits and as a result unit objects produced.
    The basic idea is to have the 'base_unit' class that contains 7 fundamental SI dimensions: mass (kg), length (m), 
    time (s),... and a multiplier used for unit conversions. It defines math. operators *, /, ** to ease creation of derived units.
    The 'base_unit' class is used by the 'unit' class to define a set of derived units (like Volt, Joule, Watt ...) 
    that can be used to specify units of quantities. 'unit' class also defines math. operators *, /, **.
    A term quantity describes an object that has a value and units (the 'quantity' class). It defines all mathematical and 
    logical operators, and math. functions so that the units consistency can be tested.
    Again, there is some additional info in the source code files and some tests; however it would be a good idea 
    to have it tested by more people.
    
3.  NineML component tester
    It comes in two variants:
        - Desktop application (http://nineml-webapp.incf.org)
        - Web application (some working examples can be found in nineml_tester_gui_examples.py)
    What the applications do is:
        - take as an input a NineMl component (python abstraction layer component object)
        - analyse it
        - show the GUI to the user where the values of parameters, initial conditions, analog ports inputs, event ports inputs, 
          initially active regimes and a list of variables to be reported can be entered
        - validate input data
        - run the simulation
        - prepare data necessary for the model report
        - generate Latex model report and export it to pdf
    
    Again the comments/suggestions/recommendations would be very useful.
    