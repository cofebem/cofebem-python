SetFactory("OpenCASCADE");

// ---- geometric parameters ----
L  = 40;          // length  (x)
W  = 40;          // width   (y)
H  = 20;          // height  (z)
R  = 5;           // radius of main indentation

// ---- mesh parameters ---------
hMin = 0.5;      // finest size (near indentation)
hMed = 2.00;      // intermediate size 
hMax = 5.00;      // coarsest size (far field)

// Refinement zone radii
R_fine = R;       // fine mesh zone radius
R_med = 2*R;      // intermediate mesh zone radius

// ---- Box geometry ----
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

// ---- Main indentation (semi-sphere) ----
// Center point for indentation
Point(101) = {L/2, W/2, H, hMin};

// Points for fine mesh refinement around indentation
Point(102) = {L/2+R_fine, W/2, H, hMin};
Point(103) = {L/2-R_fine, W/2, H, hMin};
Point(104) = {L/2, W/2+R_fine, H, hMin};
Point(105) = {L/2, W/2-R_fine, H, hMin};
Point(106) = {L/2, W/2, H-R_fine, hMin};

// Points for intermediate mesh refinement 
Point(107) = {L/2+R_med, W/2, H, hMed};
Point(108) = {L/2-R_med, W/2, H, hMed};
Point(109) = {L/2, W/2+R_med, H, hMed};
Point(110) = {L/2, W/2-R_med, H, hMed};
Point(111) = {L/2, W/2, H-R_med, hMed};

// Additional intermediate zone points for better transition
Point(112) = {L/2+R_med*0.7, W/2+R_med*0.7, H, hMed};
Point(113) = {L/2-R_med*0.7, W/2+R_med*0.7, H, hMed};
Point(114) = {L/2+R_med*0.7, W/2-R_med*0.7, H, hMed};
Point(115) = {L/2-R_med*0.7, W/2-R_med*0.7, H, hMed};

// Create the main semi-spherical indentation
Sphere(1) = {L/2., W/2., H, R, -Pi/2, 0., 2*Pi};

// ---- Box faces ----
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

// Top surface with indentation
Curve Loop(8) = {3, 12, -7, -11};
Curve Loop(9) = {13};  // Curve from sphere
Plane Surface(8) = {8, 9};

// Volume definition
Curve Loop(10) = {13};
Surface Loop(2) = {1, 8, 4, 3, 6, 5, 7};
Volume(2) = {2};

// ---- Mesh size fields for multi-level refinement ----

// Field 1: Fine mesh near indentation center
Field[1] = Distance;
Field[1].PointsList = {101};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = hMin;
Field[2].SizeMax = hMed;
Field[2].DistMin = R_fine;
Field[2].DistMax = R_fine * 1.2;

// Field 3: Intermediate mesh in medium zone
Field[3] = Distance;
Field[3].PointsList = {101};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = hMed;
Field[4].SizeMax = hMax;
Field[4].DistMin = R_med;
Field[4].DistMax = R_med * 1.5;

// Field 5: Combined field using minimum of the two zones
Field[5] = Min;
Field[5].FieldsList = {2, 4};

// Field 6: Background field for smooth transition
Field[6] = Distance;
Field[6].CurvesList = {13}; // Distance from indentation curve

Field[7] = Threshold;
Field[7].InField = 6;
Field[7].SizeMin = hMin;
Field[7].SizeMax = hMax;
Field[7].DistMin = 0;
Field[7].DistMax = R_med * 2;

// Final combined field
Field[8] = Min;
Field[8].FieldsList = {5, 7};

Background Field = 8;

// ---- Mesh options ----
Mesh.Algorithm = 6;           // Frontal-Delaunay
Mesh.Algorithm3D = 1;         // Delaunay
Mesh.CharacteristicLengthMin = hMin;
Mesh.CharacteristicLengthMax = hMax;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;