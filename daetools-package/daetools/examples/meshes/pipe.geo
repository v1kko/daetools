lc = 1;
xc = 0;
yc = 0;
r = 0.2;
R = 0.6;

Point(0) = {xc, yc, 0, lc};
Point(1) = {-r, yc, 0, lc};
Point(2) = {-R, yc, 0, lc};
Point(3) = { r, yc, 0, lc};
Point(4) = { R, yc, 0, lc};
Point(5) = {xc, r, 0, lc};
Point(6) = {xc, R, 0, lc};
Point(7) = {xc, -r, 0, lc};
Point(8) = {xc, -R, 0, lc};

Line(10) = {1, 2};
Line(20) = {3, 4};
Line(30) = {5, 6};
Line(40) = {7, 8};

Circle(1) = {2, 0, 6};
Circle(2) = {6, 0, 4};
Circle(3) = {4, 0, 8};
Circle(4) = {8, 0, 2};
Circle(5) = {1, 0, 5};
Circle(6) = {5, 0, 3};
Circle(7) = {3, 0, 7};
Circle(8) = {7, 0, 1};

Line Loop(100) = {1, 2, 3, 4};
Line Loop(200) = {5, 6, 7, 8};

Line Loop(201) = {1, 2, -20, -6, -5, 10};
Line Loop(203) = {4, -10, -8, -7, 20, 3};

Plane Surface(202) = {201};
Plane Surface(204) = {203};

Transfinite Line{1} = 20 Using Progression 1;
Transfinite Line{2} = 20 Using Progression 1;
Transfinite Line{3} = 20 Using Progression 1;
Transfinite Line{4} = 20 Using Progression 1;
Transfinite Line{5} = 20 Using Progression 1;
Transfinite Line{6} = 20 Using Progression 1;
Transfinite Line{7} = 20 Using Progression 1;
Transfinite Line{8} = 20 Using Progression 1;

Transfinite Line{10} = 20 Using Bump 0.05;
Transfinite Line{20} = 20 Using Bump 0.05;
Transfinite Line{30} = 20 Using Bump 0.05;
Transfinite Line{40} = 20 Using Bump 0.05;

Transfinite Surface {202} = {1, 2, 3, 4};
Transfinite Surface {204} = {1, 2, 3, 4};

Recombine Surface{202};
Recombine Surface{204};

Physical Line(0) = {1, 2};
Physical Line(1) = {3, 4};
Physical Line(2) = {5, 6, 7, 8};

Physical Surface(1) = 202;
Physical Surface(2) = 204;

//Field[1] = BoundaryLayer;
//Field[1].EdgesList = {1};
//Field[1].hfar = 0.05;
//Field[1].hwall_t = 0.010;
//Field[1].thickness = 0.05;
//Background Field = 1;
