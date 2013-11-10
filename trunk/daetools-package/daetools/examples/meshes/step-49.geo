cl1 = 1;

Point(1) = {-1, 0.3, 0, 1};
Point(2) = {0.5, 0.3, 0, 1};
Point(3) = {-1, -0.5, 0, 1};
Point(4) = {0.5, -0.5, 0, 1};

Point(7) = {-0.3, -0.1, 0, 1};
Point(8) = {-0.2, -0.1, 0, 1};
Point(9) = {-0.3, 0.1, -0, 1};
Point(10) = {-0.4, -0.1, 0, 1};
Point(11) = {-0.3, -0.3, 0, 1};

Point(12) = {0.1, -0.1, 0, 1};
Point(13) = {0.2, 0.0, 0, 1};
Point(14) = {0.3, -0.1, 0, 1};

Point(209) = {-0.3, 0.3, 0, 1};
Point(211) = {-0.3, -0.5, 0, 1};
Point(213) = {0.2, 0.3, 0, 1};
Point(214) = {0.2, -0.2, 0, 1};
Point(215) = {0.2, -0.5, 0, 1};

// lines of the outer box:
Line(1) = {1, 209};
Line(2) = {209, 213};
Line(3) = {213, 2};
Line(4) = {2, 4};
Line(5) = {4, 215};
Line(6) = {215, 211};
Line(7) = {211, 3};
Line(8) = {3, 1};

// the first cutout:
Ellipse(51) = {8, 7, 11, 9};
Ellipse(61) = {9, 7, 11, 10};
Ellipse(71) = {8, 7, 10, 11};
Ellipse(81) = {11, 7, 8, 10};

// the second cutout:
Line(91) = {12, 13};
Line(101) = {13, 14};
Line(111) = {12, 214};
Line(121) = {214, 14};

Line(21) = {9, 209};
Line(22) = {11, 211};
Line(23) = {13, 213};
Line(24) = {214, 215};

Transfinite Line{1} = 30 Using Progression 1;
Transfinite Line{2} = 20 Using Progression 1;
Transfinite Line{3} = 10 Using Progression 1;
Transfinite Line{4} = 47 Using Progression 1;
Transfinite Line{5} = 10 Using Progression 1;
Transfinite Line{6} = 20 Using Progression 1;
Transfinite Line{7} = 30 Using Progression 1;
Transfinite Line{8} = 47 Using Progression 1;

Transfinite Line{51} = 15 Using Progression 1;
Transfinite Line{61} = 15 Using Progression 1;
Transfinite Line{71} = 15 Using Progression 1;
Transfinite Line{81} = 15 Using Progression 1;

Transfinite Line{91} = 8 Using Progression 1;
Transfinite Line{101} = 8 Using Progression 1;
Transfinite Line{111} = 8 Using Progression 1;
Transfinite Line{121} = 8 Using Progression 1;

Transfinite Line{21} = 10 Using Progression 1;
Transfinite Line{22} = 10 Using Progression 1;
Transfinite Line{23} = 17 Using Progression 1;
Transfinite Line{24} = 17 Using Progression 1;

// these define the boundary indicators in deal.II:
Physical Line(0) = {1, 2, 4, 3};
Physical Line(1) = {61, 51, 81, 71};
Physical Line(2) = {91, 101, 111, 121};

Line Loop(112) = {7, 8, 1, -21, 61, -81, 22};
Plane Surface(113) = {112};
Line Loop(122) = {2, -23, -91, 111, 24, 6, -22, -71, 51, 21};
Plane Surface(123) = {122};
Line Loop(124) = {3, 4, 5, -24, 121, -101, 23};
Plane Surface(125) = {124};

Recombine Surface{113};
Recombine Surface{123};
Recombine Surface{125};

Transfinite Surface {113} = {1, 3, 209, 211};
Transfinite Surface {123} = {209, 211, 213, 215};
Transfinite Surface {125} = {2, 4, 213, 215};

Physical Surface(0) = {113};
Physical Surface(1) = {123};
Physical Surface(2) = {125};

//Field[1] = BoundaryLayer;
//Field[1].EdgesList = {51,61,71,81};
//Field[1].NodesList = {8,9,10,11};
//Field[1].hfar = 0.05;
//Field[1].hwall_n = 0.010;
//Field[1].thickness = 0.05;
//Background Field = 1;

// some parameters for the meshing:
//Mesh.Algorithm = 8;
//Mesh.RecombineAll = 1;
//Mesh.CharacteristicLengthFactor = 0.05;
//Mesh.SubdivisionAlgorithm = 1;
//Mesh.Smoothing = 80;
//Show "*";
