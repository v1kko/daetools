Point(1) = {  0, 0, 0, 5};
Point(2) = {100, 0, 0, 5};

// lines of the outer box:
Line(1) = {1, 2};

// these define the boundary indicators in deal.II:
Physical Point(0) = {1};
Physical Point(1) = {2};
Physical Line(0)  = {1};
