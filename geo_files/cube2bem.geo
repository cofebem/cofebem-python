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
Mesh.MshFileVersion = 2.2;
Mesh.SaveAll        = 1;     
Mesh.RecombineAll = 0;

// -------- Geometry --------
Box(1) = {0, 0, 0, L, L, L}; // volume id = 1

// Name faces via bounding boxes
left[]   = Surface In BoundingBox(-1e-12, 0,       0,      1e-12,   L,       L);
right[]  = Surface In BoundingBox( L-1e-12,0,       0,      L+1e-12, L,       L);
front[]  = Surface In BoundingBox( 0,      -1e-12,  0,      L,       1e-12,   L);
back[]   = Surface In BoundingBox( 0,       L-1e-12,0,      L,       L+1e-12, L);
bottom[] = Surface In BoundingBox( 0,       0,     -1e-12,  L,       L,       1e-12);
top[]    = Surface In BoundingBox( 0,       0,      L-1e-12,L,       L,       L+1e-12);

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

// -------- Refine only the top face edges --------
TopEdges[] = Curve In BoundingBox(0, 0, L-1e-12, L, L, L+1e-12);
MeshSize{ TopEdges[] } = lc/10;  // refine only edges of the top surface
