SetFactory("OpenCASCADE");

// ---------------------------
// Parameters
// ---------------------------
R = 1.0;

lc_far = R/8;     // coarse
lc_tip = R/80;    // fine at the pole
d_max  = 0.5*R;  // grading distance

// ---------------------------
// Mesh options
// ---------------------------
Mesh.Algorithm3D    = 4;
Mesh.ElementOrder   = 1;
Mesh.MshFileVersion = 2.2;

Mesh.CharacteristicLengthMin = lc_tip;
Mesh.CharacteristicLengthMax = lc_far;

// Make the background field drive the 3D mesh
Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.MeshSizeFromCurvature      = 0;

// ---------------------------
// Geometry: hemisphere = sphere cut by plane z=0
// ---------------------------
Sphere(1) = {0, 0, 0, R};

// Half-space box (z>=0)
eps = 1e-6;
Box(2) = {-R-1e-3, -R-1e-3, 0,
          2*R+2e-3, 2*R+2e-3, R+1e-3};

hem[] = BooleanIntersection{ Volume{1}; Delete; }{ Volume{2}; Delete; };
hemVol = hem[0];

// ---------------------------
// IMPORTANT: create a REAL CAD point at the top pole and imprint it
// ---------------------------
pTop = 1000;
Point(pTop) = {0, 0, R, lc_tip};

// Imprint point into the volume boundary so it becomes a mesh constraint
tmp[] = BooleanFragments{ Volume{hemVol}; Delete; }{ Point{pTop}; Delete; };
hemVol = tmp[0];

// ---------------------------
// Get boundary surfaces (robust way)
// ---------------------------
allSurf[] = Boundary{ Volume{hemVol}; };

baseSurf[] = Surface In BoundingBox(-R-1e-3, -R-1e-3, -1e-3,
                                    R+1e-3,  R+1e-3,  1e-3);

curvedSurf[] = {};
For i In {0:#allSurf[]-1}
  If(!InList(baseSurf[], allSurf[i]))
    curvedSurf[] += {allSurf[i]};
  EndIf
EndFor

// ---------------------------
// Mesh field: distance to the (now imprinted) top pole point
// ---------------------------
Field[1] = Distance;
Field[1].NodesList = {pTop};

Field[2] = Threshold;
Field[2].IField  = 1;
Field[2].LcMin   = lc_tip;
Field[2].LcMax   = lc_far;
Field[2].DistMin = 0;
Field[2].DistMax = d_max;

Background Field = 2;

// ---------------------------
// Physical groups
// ---------------------------
Physical Volume("Hemisphere")        = {hemVol};
Physical Surface("Hemisphere_Base")  = {baseSurf[]};
Physical Surface("Hemisphere_Shell") = {curvedSurf[]};
