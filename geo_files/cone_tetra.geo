SetFactory("OpenCASCADE");

// ---------------- Parameters ----------------
x0 = 0.0;
y0 = 0.0;
z0 = 0.0;      // base center (top of cone)
R  = 1.0;      // base radius
H  = 1.0;      // height (apex at z0 - H)

lc = 0.08;
order = 1;

// ---------------- Mesh options ----------------
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.ElementOrder = order;
Mesh.Optimize = 1;
Mesh.HighOrderOptimize = 1;

// ---------------- Geometry ----------------
// Axis points downward → dz = -H
// Apex will be at z = z0 - H
Cone(1) = {x0, y0, z0, 0, 0, -H, R, 0};

// ---------------- Physical groups ----------------
Physical Volume("Cone") = {1};

// Boundary tagging
cone_surfs[] = Boundary{ Volume{1}; };
Physical Surface("ConeBoundary") = {cone_surfs[]};

// Identify base (top disk) and lateral surface
eps = 1e-9;

base[] = Surface In BoundingBox(
    x0-R-1e-6, y0-R-1e-6, z0-eps,
    x0+R+1e-6, y0+R+1e-6, z0+eps
);

lateral[] = Surface In BoundingBox(
    x0-R-1e-6, y0-R-1e-6, z0-H-1e-6,
    x0+R+1e-6, y0+R+1e-6, z0-eps
);

Physical Surface("ConeBaseTop") = {base[]};
Physical Surface("ConeLateral") = {lateral[]};
