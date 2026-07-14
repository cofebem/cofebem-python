//+
lc = 30.;
//+
Point(1) = {0, 0, 0, lc};
//+
Point(2) = {0, 228.6, 0, lc};
//+
Point(3) = {0, 352, 0, lc};
//+
Point(4) = {0, 357, 0, lc};
//+
Point(5) = {0, 360, 0, lc};
//+
Point(6) = {56.4, 360, 0, lc};
//+
Point(7) = {56.4, 357, 0, lc};
//+
Point(8) = {56.4, 352, 0, lc};
//+
Point(9) = {56.4, 310.6, 0, lc};
//+
Point(10) = {146.9, 320.9, 0, lc};
//+
Point(11) = {144, 320.2, 0, lc};
//+
Point(12) = {139.1, 319, 0, lc};
//+
Point(13) = {12.6, 266.5, 0, lc};
//+
Point(14) = {152.5, 228.6, 0, lc};
//+
Point(15) = {149.5, 228.6, 0, lc};
//+
Point(16) = {144.5, 228.6, 0, lc};
//+
Point(17) = {148.8, 310.6, 0, lc};
//+
Point(18) = {145.8, 310.6, 0, lc};
//+
Point(19) = {140.8, 310.6, 0, lc};


//+
Line(1) = {3, 4};
//+
Line(2) = {4, 5};
//+
Line(3) = {5, 6};
//+
Line(4) = {4, 7};
//+
Line(5) = {3, 8};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {14, 15};
//+
Line(9) = {15, 16};
//+
Ellipse(10) = {10, 9, 17, 6};
//+
Ellipse(11) = {11, 9, 18, 7};
//+
Ellipse(12) = {12, 9, 19, 8};
//+
Circle(13) = {10, 13, 14};
//+
Circle(14) = {11, 13, 15};
//+
Circle(15) = {12, 13, 16};

//+
Line Loop(16) = {1, 4, 7, -5};
//+ make surface
Plane Surface(17) = {16};
//+
Line Loop(18) = {2, 3, 6, -4};
//+ make surface
Plane Surface(19) = {18};
//+
Curve Loop(20) = {-6, -10, 13, 8, -14, 11};
//+ make surface
Plane Surface(21) = {20};
//+
Curve Loop(22) = {-7, -11, 14, 9, -15, 12};
//+ make surface
Plane Surface(23) = {22};
//+
// Duplicate the surfaces
MirroredSurfaces[] = Rotate {{0,1,0}, {0,0,0}, Pi} { Duplicata{ Surface{17}; Surface{23}; } };
Printf("New surfaces '%g' and '%g'", MirroredSurfaces[0], MirroredSurfaces[1]);
//
el1[] = Extrude { {1,0,0}, {0,0,0}, .999*Pi} {
    Surface{17};
    Surface{23};
    Surface{MirroredSurfaces[0]};
    Surface{MirroredSurfaces[1]};
};
//+ Define the physical groups
MirroredSurfaces[] = Rotate {{0,1,0}, {0,0,0}, Pi} { Duplicata{ Surface{19}; Surface{21}; } };
Printf("New surfaces '%g' and '%g'", MirroredSurfaces[0], MirroredSurfaces[1]);
//
el2[] = Extrude { {1,0,0}, {0,0,0}, .999*Pi} {
    Surface{19};
    Surface{21};
    Surface{MirroredSurfaces[0]};
    Surface{MirroredSurfaces[1]};
};
//+ Define the physical groups
Physical Volume(1004) = {1,2,3,4};
Physical Volume(1002) = {5,6,7,8};
Physical Surface(1011) = {84, 88, 56, 110, 142, 138};
Physical Surface(1123) = {80, 198, 134, 252};
//+
// Define the MathEval field with a custom function
Field[1] = MathEval;
Field[1].F = "0.5 * (25. - 22.5 / (1 + (x/150)^2 + (y/150)^2 + ((360-z)/150)^2))"; // Example function
// Field[1].F = "25. - 22.5 / (1 + (x/150)^2 + (y/150)^2 + ((360-z)/150)^2)"; // Example function
// Set this field as the background field
Background Field = 1;
