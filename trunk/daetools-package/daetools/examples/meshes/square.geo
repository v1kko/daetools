cl1 = 1;

Point(1) = {0, 0, 0, 1};
Point(2) = {0, 1, 0, 1};
Point(3) = {1, 1, 0, 1};
Point(4) = {1, 0, 0, 1};

// lines of the outer box:
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Transfinite Line{1} = 20;//Using Progression 1;
Transfinite Line{2} = 20;
Transfinite Line{3} = 20;
Transfinite Line{4} = 20;

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

