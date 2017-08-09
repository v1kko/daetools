x1 = 0.00e-3;
x2 = 0.06e-3;
x3 = 0.10e-3;

y1 = -0.1e-3;
y2 =  0.0e-3;
y3 =  0.1e-3;
y4 =  0.2e-3;

N1 = 10;
N2 = 30;
N3 = 15;
N4 = 30;
N5 = 15;

Macro ActiveSurface

    Point(7)  = {y2, 0, 0, 1};
    Point(8)  = {y3, 0, 0, 1};

    Line(8) = {7, 8};
    Transfinite Line{8} = N4-2;
    // Active surface line
    Physical Point(0) = {7};
    Physical Point(1) = {8};
    
    Physical Line(5) = {8};
    
Return

Macro ParallelPlateReactor

    Point(1)  = {x1, y1, 0, 1};
    Point(2)  = {x1, y2, 0, 1};
    Point(3)  = {x1, y3, 0, 1};
    Point(4)  = {x1, y4, 0, 1};

    Point(10) = {x2, y1, 0, 1};
    Point(11) = {x2, y2, 0, 1};
    Point(12) = {x2, y3, 0, 1};
    Point(5)  = {x2, y4, 0, 1};

    Point(6)  = {x3, y4, 0, 1};
    Point(7)  = {x3, y3, 0, 1};
    Point(8)  = {x3, y2, 0, 1};
    Point(9)  = {x3, y1, 0, 1};

    // lines:
    Line(1) = {1, 10};
    Line(2) = {1, 2};
    Line(3) = {2, 3};
    Line(4) = {3, 4};
    Line(5) = {4, 5};
    Line(6) = {5, 6};
    Line(7) = {6, 7};
    Line(8) = {7, 8};
    Line(9) = {8, 9};
    Line(10)= {9, 10};
    Line(11)= {2, 11};
    Line(12)= {3, 12};
    Line(13)= {12, 7};
    Line(14)= {11, 8};
    Line(15)= {5, 12};
    Line(16)= {12, 11};
    Line(17)= {11, 10};

    Transfinite Line{1}  = N1;
    Transfinite Line{2}  = N3;
    Transfinite Line{3}  = N4;
    Transfinite Line{4}  = N5;
    Transfinite Line{5}  = N1;
    Transfinite Line{6}  = N2;
    Transfinite Line{7}  = N5;
    Transfinite Line{8}  = N4;
    Transfinite Line{9}  = N3;
    Transfinite Line{10} = N2;

    // these define the boundary indicators in deal.II:
    Physical Line(0) = {2, 3, 4};
    Physical Line(1) = {5, 6};
    Physical Line(2) = {7, 9};
    Physical Line(3) = {1, 10};
    // Active surface line
    Physical Line(5) = {8};

    Line Loop(112) = {-1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Plane Surface(113) = {112};
    Transfinite Surface {113} = {1, 4, 6, 9};
    Recombine Surface(113); // combine triangles into quads

    Physical Surface(0) = {113};
Return

Call ParallelPlateReactor;

//Call ActiveSurface;
