********************
Programming language
********************
..
    Copyright (C) Dragan Nikolic, 2013
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

**DAE Tools** core libraries are written in standard c++. However, `Python <http://www.python.org>`_ programming language is
used as the main modelling language. The main reason for use of Python is (as the authors say):
"*Python is an easy to learn, powerful programming language. It has efficient high-level data structures and a simple
but effective approach to object-oriented programming. Python's elegant syntax and dynamic typing, together with its
interpreted nature, make it an ideal language for scripting and rapid application development in many areas on
most platforms*" `link <http://docs.python.org/tutorial>`_.

And: *"Often, programmers fall in love with Python because of the increased productivity it provides. Since there is no
compilation step, the edit-test-debug cycle is incredibly fast*" `link <http://www.python.org/doc/essays/blurb>`_. Also, please
have a look on `a comparison to the other languages <http://www.python.org/doc/essays/comparisons>`_. Based on the information
available online, and according to the personal experience, the python programs are much shorter and take an order of magnitude
less time to develop it. Initially I developed daePlotter module in c++; it took me about one month of part time coding. But,
then I moved to python: reimplementing it in PyQt took me just two days (with several new features added), while the code size
shrank from 24 cpp modules to four python modules only!

"*Where Python code is typically 3-5 times shorter than equivalent Java code, it is often 5-10 times shorter than equivalent
C++ code! Anecdotal evidence suggests that one Python programmer can finish in two months what two C++ programmers can't
complete in a year. Python shines as a glue language, used to combine components written in C++*"
`link <http://www.python.org/doc/essays/comparisons>`_.
Obviously, not everything can be developed in python; a heavy c++ artillery is still necessary for highly complex projects.



.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:
