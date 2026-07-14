SetFactory("OpenCASCADE");

// ---------------- Parameters ----------------
Rc = 1.0;      // radius
Hc = 1.0;       // height (+z direction)

x0 = 0.0;
y0 = 0.0;
zc0 = -Hc/2;      // bottom z position


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
// Cylinder(tag) = {x, y, z, dx, dy, dz, r}
Cylinder(1) = {x0, y0, zc0, 0, 0, Hc, Rc};

// ---------------- Physical groups ----------------
Physical Volume("Cylinder") = {1};

// Boundary tagging
all_surfs[] = Boundary{ Volume{1}; };
Physical Surface("CylinderBoundary") = {all_surfs[]};

// Identify specific surfaces
eps = 1e-9;

bottom[] = Surface In BoundingBox(
    x0-Rc-1e-6, y0-Rc-1e-6, zc0-eps,
    x0+Rc+1e-6, y0+Rc+1e-6, zc0+eps
);

top[] = Surface In BoundingBox(
    x0-Rc-1e-6, y0-Rc-1e-6, zc0+Hc-eps,
    x0+Rc+1e-6, y0+Rc+1e-6, zc0+Hc+eps
);

side[] = Surface In BoundingBox(
    x0-Rc-1e-6, y0-Rc-1e-6, zc0+eps,
    x0+Rc+1e-6, y0+Rc+1e-6, zc0+Hc-eps
);

Physical Surface("CylinderBottom") = {bottom[]};
Physical Surface("CylinderTop") = {top[]};
Physical Surface("CylinderLateral") = {side[]};
