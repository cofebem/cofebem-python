SetFactory("OpenCASCADE");

// ---------------- Parameters ----------------
L      = 1.0;
lc     = L/8;
lc_top = lc/5;   
order  = 1;
t      = L/20;   // thickness of the refinement "tube" around each edge

// ---------------- Mesh options ----------------
Mesh.Algorithm3D       = 4;
Mesh.ElementOrder      = order;
Mesh.Optimize          = 1;
Mesh.HighOrderOptimize = 1;

// IMPORTANT for dolfinx
//Mesh.MshFileVersion = 4.1;

Mesh.SaveAll        = 0;
Mesh.RecombineAll   = 0;

// Let fields dominate
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

Mesh.CharacteristicLengthMin = lc_top/2;
Mesh.CharacteristicLengthMax = lc;

// ---------------- Geometry ----------------
Box(1) = {0, 0, 0, L, L, L};

// ---------------- Physical groups ----------------
Physical Volume("Cube") = {1};
// Get all boundary surfaces of the volume
allBoundarySurfaces[] = Boundary{ Volume{1}; };
Physical Surface("Boundary") = {allBoundarySurfaces[]};

// Tag faces (recommended)
//epsS = 1e-9 * L;

//leftS[]   = Surface In BoundingBox(-epsS, -epsS, -epsS,      epsS, L+epsS, L+epsS);
//rightS[]  = Surface In BoundingBox(L-epsS, -epsS, -epsS,  L+epsS, L+epsS, L+epsS);
//frontS[]  = Surface In BoundingBox(-epsS, -epsS, -epsS,  L+epsS,     epsS, L+epsS);
//backS[]   = Surface In BoundingBox(-epsS, L-epsS, -epsS,  L+epsS,  L+epsS, L+epsS);
//bottomS[] = Surface In BoundingBox(-epsS, -epsS, -epsS,  L+epsS,  L+epsS,     epsS);
//topS[]    = Surface In BoundingBox(-epsS, -epsS, L-epsS,  L+epsS,  L+epsS,  L+epsS);

//Physical Surface("Left")   = {leftS[]};
//Physical Surface("Right")  = {rightS[]};
//Physical Surface("Front")  = {frontS[]};
//Physical Surface("Back")   = {backS[]};
//Physical Surface("Bottom") = {bottomS[]};
//Physical Surface("Top")    = {topS[]};

// ---------------- Size fields: refine ALL 12 edges ----------------
// Convention: each Box defines a "tube" of thickness t around one edge.
// VIn is the fine size, VOut is the coarse size.

Field[1] = Box;  // x-edge: y~0, z~0
Field[1].VIn=lc_top; Field[1].VOut=lc;
Field[1].XMin=0; Field[1].XMax=L;
Field[1].YMin=0; Field[1].YMax=t;
Field[1].ZMin=0; Field[1].ZMax=t;

Field[2] = Box;  // x-edge: y~L, z~0
Field[2].VIn=lc_top; Field[2].VOut=lc;
Field[2].XMin=0; Field[2].XMax=L;
Field[2].YMin=L-t; Field[2].YMax=L;
Field[2].ZMin=0; Field[2].ZMax=t;

Field[3] = Box;  // x-edge: y~0, z~L
Field[3].VIn=lc_top; Field[3].VOut=lc;
Field[3].XMin=0; Field[3].XMax=L;
Field[3].YMin=0; Field[3].YMax=t;
Field[3].ZMin=L-t; Field[3].ZMax=L;

Field[4] = Box;  // x-edge: y~L, z~L
Field[4].VIn=lc_top; Field[4].VOut=lc;
Field[4].XMin=0; Field[4].XMax=L;
Field[4].YMin=L-t; Field[4].YMax=L;
Field[4].ZMin=L-t; Field[4].ZMax=L;

Field[5] = Box;  // y-edge: x~0, z~0
Field[5].VIn=lc_top; Field[5].VOut=lc;
Field[5].XMin=0; Field[5].XMax=t;
Field[5].YMin=0; Field[5].YMax=L;
Field[5].ZMin=0; Field[5].ZMax=t;

Field[6] = Box;  // y-edge: x~L, z~0
Field[6].VIn=lc_top; Field[6].VOut=lc;
Field[6].XMin=L-t; Field[6].XMax=L;
Field[6].YMin=0; Field[6].YMax=L;
Field[6].ZMin=0; Field[6].ZMax=t;

Field[7] = Box;  // y-edge: x~0, z~L
Field[7].VIn=lc_top; Field[7].VOut=lc;
Field[7].XMin=0; Field[7].XMax=t;
Field[7].YMin=0; Field[7].YMax=L;
Field[7].ZMin=L-t; Field[7].ZMax=L;

Field[8] = Box;  // y-edge: x~L, z~L
Field[8].VIn=lc_top; Field[8].VOut=lc;
Field[8].XMin=L-t; Field[8].XMax=L;
Field[8].YMin=0; Field[8].YMax=L;
Field[8].ZMin=L-t; Field[8].ZMax=L;

Field[9] = Box;  // z-edge: x~0, y~0
Field[9].VIn=lc_top; Field[9].VOut=lc;
Field[9].XMin=0; Field[9].XMax=t;
Field[9].YMin=0; Field[9].YMax=t;
Field[9].ZMin=0; Field[9].ZMax=L;

Field[10] = Box; // z-edge: x~L, y~0
Field[10].VIn=lc_top; Field[10].VOut=lc;
Field[10].XMin=L-t; Field[10].XMax=L;
Field[10].YMin=0; Field[10].YMax=t;
Field[10].ZMin=0; Field[10].ZMax=L;

Field[11] = Box; // z-edge: x~0, y~L
Field[11].VIn=lc_top; Field[11].VOut=lc;
Field[11].XMin=0; Field[11].XMax=t;
Field[11].YMin=L-t; Field[11].YMax=L;
Field[11].ZMin=0; Field[11].ZMax=L;

Field[12] = Box; // z-edge: x~L, y~L
Field[12].VIn=lc_top; Field[12].VOut=lc;
Field[12].XMin=L-t; Field[12].XMax=L;
Field[12].YMin=L-t; Field[12].YMax=L;
Field[12].ZMin=0; Field[12].ZMax=L;

// Combine all edge refinements
Field[13] = Min;
Field[13].FieldsList = {1,2,3,4,5,6,7,8,9,10,11,12};

Background Field = 13;

// Ensure we actually generate a 3D mesh
Mesh 3;
