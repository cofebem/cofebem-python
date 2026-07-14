SetFactory("OpenCASCADE");

// ---------------- Parameters ----------------
x0 = 0.0;
y0 = 0.0;

Lg = 5.0;        // size in x and y
Tg = 0.03;       // thickness
zg_top = 0.0;    // top surface (contact plane)

// Mesh
lc = 0.08;
order = 1;

// ---------------- Mesh options ----------------
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.ElementOrder = order;
Mesh.Optimize = 1;
Mesh.HighOrderOptimize = 1;

// ---------------- Geometry ----------------
// Top face at zg_top → bottom at zg_top - Tg
Box(1) = {-Lg/2, -Lg/2, zg_top - Tg, Lg, Lg, Tg};

// ---------------- Physical groups ----------------
Physical Volume("Ground") = {1};

// Boundary surfaces
all_surfs[] = Boundary{ Volume{1}; };
Physical Surface("GroundBoundary") = {all_surfs[]};

// Contact surface (top face)
eps = 1e-9;
top[] = Surface In BoundingBox(
    -Lg/2 - 1e-6, -Lg/2 - 1e-6, zg_top - eps,
     Lg/2 + 1e-6,  Lg/2 + 1e-6, zg_top + eps
);

Physical Surface("GroundTop") = {top[]};
