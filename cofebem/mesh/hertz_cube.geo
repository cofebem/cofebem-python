SetFactory("OpenCASCADE");

// ---- geometry ----
L = 40;  W = 40;  H = 20;
Box(1) = {0, 0, 0,  L, W, H};

// ---- local-refinement field (unchanged) ----
hMin = 0.20;   hMax = 5.0;
rMin = 2;      rMax = 10;
Point(100) = {L/2, W/2, H, hMin};

Field[1] = Attractor;   Field[1].PointsList = {100};
Field[2] = Threshold;
Field[2].IField   = 1;
Field[2].LcMin    = hMin;
Field[2].LcMax    = hMax;
Field[2].DistMin  = rMin;
Field[2].DistMax  = rMax;
Background Field = 2;

// ---- produce *linear* tetrahedra only ----
Mesh.RecombineAll      = 0;     // ← no quads ⇒ no hex recombination
Mesh.Algorithm3D       = 4;     // Frontal-Delaunay tet mesher
Mesh.ElementOrder      = 1;     // first-order (4-node) tets
Mesh.Optimize          = 1;
Mesh.CharacteristicLengthMin = hMin;
Mesh.CharacteristicLengthMax = hMax;

// ---- generate & save ----
Mesh 3;
Save "hertz_cube.vtk";    // Gmsh 2.2 ASCII, perfect for dolfinx
