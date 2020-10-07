Point(1) = {0, 2, 0, 1};
Point(2) = {2, 2, 0, 1};
Point(3) = {4, 0, 0, 1};
Point(4) = {2, 0, 0, 1};
Point(5) = {0, 0, 0, 1};

// lines of the outer box:
Line(1) = {1, 2};
Line(2) = {2, 4};
Line(3) = {4, 3};
Line(4) = {4, 5};
Line(5) = {5, 1};
Circle(10) = {2, 4, 3};

Line Loop(100) = {1, 10, -3, 4, 5};

//Transfinite Line{1} = 21;
//Transfinite Line{2} = 21;
//Transfinite Line{3} = 21;
//Transfinite Line{4} = 21;
//Transfinite Line{5} = 21;
//Transfinite Line{10} = 21;

// these define the boundary indicators in deal.II:
Physical Line(1) = {5, 1, 10};
Physical Line(2) = {3,4};

Plane Surface(200) = {100};
Physical Surface(200) = 200;

//Transfinite Surface {200} = {1, 2, 4, 5};
Recombine Surface{200};
