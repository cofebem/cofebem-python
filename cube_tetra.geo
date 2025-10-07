SetFactory("OpenCASCADE");

// -------- Parameters --------
L  = 1.0;      // cube edge
lc = L/8;      // target size
order = 1;     // 1 or 2 (P1/P2)

// -------- Mesh options --------
Mesh.Algorithm3D = 4;       // Delaunay
Mesh.ElementOrder = order;
Mesh.Optimize = 1;
Mesh.HighOrderOptimize = 1;
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.MshFileVersion = 2.2;          //4.1;   // <- use MSH4.1 for dolfinx  and 2.2 for meshio
Mesh.SaveAll        = 1;     
Mesh.RecombineAll = 0;      // ensure triangles on surfaces

// -------- Geometry --------
Box(1) = {0, 0, 0, L, L, L}; // volume id = 1

// Name faces via bounding boxes (robust & readable)
left[]   = Surface In BoundingBox(-1e-12, 0,       0,        1e-12,   L,       L); // x=0
right[]  = Surface In BoundingBox( L-1e-12,0,       0,        L+1e-12, L,       L); // x=L
front[]  = Surface In BoundingBox( 0,      -1e-12,  0,        L,       1e-12,   L); // y=0
back[]   = Surface In BoundingBox( 0,       L-1e-12,0,        L,       L+1e-12, L); // y=L
bottom[] = Surface In BoundingBox( 0,       0,     -1e-12,    L,       L,       1e-12); // z=0
top[]    = Surface In BoundingBox( 0,       0,      L-1e-12,  L,       L,       L+1e-12); // z=L

// -------- Physical groups --------
Physical Volume("Cube")  = {1};
Physical Surface("Left")   = {left[]};
Physical Surface("Right")  = {right[]};
Physical Surface("Front")  = {front[]};
Physical Surface("Back")   = {back[]};
Physical Surface("Bottom") = {bottom[]};
Physical Surface("Top")    = {top[]};

// Optional: enforce size at CAD vertices
CornerPts[] = PointsOf{ Volume{1}; };
MeshSize{ CornerPts[] } = lc;
