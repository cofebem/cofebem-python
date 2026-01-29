SetFactory("OpenCASCADE");

// ---- geometric parameters ----
L  = 40;          // length  (x)
W  = 40;          // width   (y)
H  = 20;          // height  (z)
R  = 5;           // radius of the indentation zone
Rm = 8;           // radius of the mesh density change

// ---- mesh parameters ---------
hMin = 0.16384;      // finest size 
hMed = 2.62144;      // intermediate size
hMax = 3.93216;      // coarsest size 

Point(1) = {0,0,0,hMax};
Point(2) = {L,0,0,hMax};
Point(3) = {L,0,H,hMax};
Point(4) = {0,0,H,hMax};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Point(5) = {0,W,0,hMax};
Point(6) = {L,W,0,hMax};
Point(7) = {L,W,H,hMax};
Point(8) = {0,W,H,hMax};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};


Sphere(1) = {L/2., W/2., H, R, -Pi/2, 0., 2*Pi};


Curve Loop(3) = {12, 8, -9, -4};
Plane Surface(3) = {3};
Curve Loop(4) = {3, 4, 1, 2};
Plane Surface(4) = {4};
Curve Loop(5) = {2, 11, -6, -10};
Plane Surface(5) = {5};
Curve Loop(6) = {6, 7, 8, 5};
Plane Surface(6) = {6};
Curve Loop(7) = {5, -10, -1, 9};
Plane Surface(7) = {7};
Curve Loop(8) = {3, 12, -7, -11};
Curve Loop(9) = {13};
Plane Surface(8) = {8, 9};
Curve Loop(10) = {13};
Surface Loop(2) = {1, 8, 4, 3, 6, 5, 7};

// Volume(2) = {2};

// Fictious points to refine mesh near the indentation zone
Point(101) = {L/2, W/2, H, hMed};
Point(102) = {L/2+R, W/2, H, hMin};
Point(103) = {L/2-R, W/2, H, hMin};
Point(104) = {L/2, W/2+R, H, hMin};
Point(105) = {L/2, W/2-R, H, hMin};
Point(106) = {L/2, W/2, H-R, hMin};
Point(1001) = {L/2+Rm, W/2, H, hMed};
Point(1002) = {L/2-Rm, W/2, H, hMed};
Point(1003) = {L/2, W/2+Rm, H, hMed};
Point(1004) = {L/2, W/2-Rm, H, hMed};
Point(1005) = {L/2, W/2, H-Rm, hMed};


Circle(101) = {1001, 101, 1005};
Circle(102) = {1002, 101, 1005};
Circle(103) = {1003, 101, 1005};
Circle(104) = {1004, 101, 1005};
Circle(105) = {1001, 101, 1003};
Circle(106) = {1003, 101, 1002};
Circle(107) = {1002, 101, 1004};
Circle(108) = {1004, 101, 1001};


//+
Curve Loop(11) = {107, 104, -102};
//+
Surface(9) = {11};
//+
Curve Loop(13) = {108, 101, -104};
//+
Surface(10) = {13};
//+
Curve Loop(15) = {105, 103, -101};
//+
Surface(11) = {15};
//+
Curve Loop(17) = {106, 102, -103};
//+
Surface(12) = {17};
//+
Recursive Delete {
  Surface{8}; 
}
//+
Curve Loop(19) = {106, 107, 108, 105};
//+
Curve Loop(20) = {13};
//+
Plane Surface(13) = {19, 20};
//+
Curve Loop(21) = {105, 106, 107, 108};
//+
Curve Loop(22) = {7, -12, -3, 11};
//+
Plane Surface(14) = {21, 22};
//+
Surface Loop(3) = {11, 12, 9, 10, 14, 5, 4, 3, 6, 7};
//+
Delete {
  Surface{14}; 
}
//+
Curve Loop(25) = {106, 107, 108, 105};
//+
Curve Loop(26) = {13};
//+
Plane Surface(16) = {25, 26};
//+
Delete {
  Surface{13}; 
}
//+
Delete {
  Surface{16}; 
}

//+
//Curve Loop(27) = {107, 108, 105, 106};
//Curve Loop(28) = {13};
//Curve Loop(29) = {27, -28};
//Plane Surface(16) = {29};
//+
Curve Loop(27) = {-13};
//+
Curve Loop(28) = {107, 108, 105, 106};
//+
Plane Surface(16) = {27, 28};
//+
Surface Loop(4) = {9, 10, 11, 12, 1, 16};
//+
Volume(2) = {4};
//+
Surface Loop(5) = {9, 10, 11, 12, 15, 6, 5, 4, 3, 7};
//+
Volume(3) = {5};

Physical Volume("cube", 1) = {1, 2, 3};