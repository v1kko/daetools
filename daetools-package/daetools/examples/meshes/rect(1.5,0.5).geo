Point(1) = {0,     0, 0, 1};
Point(2) = {0,   0.5, 0, 1};
Point(3) = {1.5, 0.5, 0, 1};
Point(4) = {1.5,   0, 0, 1};

// lines of the outer box:
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Transfinite Line{1} = 100;
Transfinite Line{2} = 200 Using Bump 0.1;
Transfinite Line{3} = 100;
Transfinite Line{4} = 200 Using Bump 0.1;

// these define the boundary indicators in deal.II:
Physical Line(0) = {1};
Physical Line(1) = {2};
Physical Line(2) = {3};
Physical Line(3) = {4};

Line Loop(112) = {1, 2, 3, 4};
Plane Surface(113) = {112};
Transfinite Surface {113} = {1, 2, 3, 4};
Recombine Surface(113); // combine triangles into quads

Physical Surface(0) = {113};

