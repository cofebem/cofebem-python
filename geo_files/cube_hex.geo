SetFactory("Built-in");

// -------------------------
// Geometrical parameters
// -------------------------

Lx = 1.0;
Ly = 1.0;
Lz = 1.0;

// Number of hexahedral elements in each direction
nx = 80;
ny = 80;
nz = 5;

// Characteristic length is irrelevant for a transfinite mesh,
// but Gmsh requires a value in the Point definitions.
lc = 1.0;

// -------------------------
// Points
// -------------------------

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {Lx,  0.0, 0.0, lc};
Point(3) = {Lx,  Ly,  0.0, lc};
Point(4) = {0.0, Ly,  0.0, lc};

Point(5) = {0.0, 0.0, Lz, lc};
Point(6) = {Lx,  0.0, Lz, lc};
Point(7) = {Lx,  Ly,  Lz, lc};
Point(8) = {0.0, Ly,  Lz, lc};

// -------------------------
// Edges
// -------------------------

// Bottom face
Line(1) = {1, 2}; // x direction
Line(2) = {2, 3}; // y direction
Line(3) = {3, 4}; // x direction
Line(4) = {4, 1}; // y direction

// Top face
Line(5) = {5, 6}; // x direction
Line(6) = {6, 7}; // y direction
Line(7) = {7, 8}; // x direction
Line(8) = {8, 5}; // y direction

// Vertical edges
Line(9)  = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// -------------------------
// Surfaces
// -------------------------

// Bottom: z = 0
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Top: z = Lz
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};

// Front: y = 0
Curve Loop(3) = {1, 10, -5, -9};
Plane Surface(3) = {3};

// Right: x = Lx
Curve Loop(4) = {2, 11, -6, -10};
Plane Surface(4) = {4};

// Back: y = Ly
Curve Loop(5) = {3, 12, -7, -11};
Plane Surface(5) = {5};

// Left: x = 0
Curve Loop(6) = {4, 9, -8, -12};
Plane Surface(6) = {6};

// -------------------------
// Volume
// -------------------------

Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// -------------------------
// Structured discretization
// -------------------------

// Gmsh expects the number of points, not the number of elements.
// Therefore, nx elements require nx + 1 points.
Transfinite Curve {1, 3, 5, 7}  = nx + 1;
Transfinite Curve {2, 4, 6, 8}  = ny + 1;
Transfinite Curve {9, 10, 11, 12} = nz + 1;

Transfinite Surface {1, 2, 3, 4, 5, 6};
Recombine Surface {1, 2, 3, 4, 5, 6};

Transfinite Volume {1};
Recombine Volume {1};

// -------------------------
// Physical groups
// -------------------------

Physical Surface("bottom", 1) = {1}; // z = 0
Physical Surface("top",    2) = {2}; // z = Lz

Physical Surface("front",  3) = {3}; // y = 0
Physical Surface("right",  4) = {4}; // x = Lx
Physical Surface("back",   5) = {5}; // y = Ly
Physical Surface("left",   6) = {6}; // x = 0

Physical Volume("domain", 1) = {1};

// -------------------------
// Mesh options
// -------------------------

Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 4.1;

// Generate the 3D mesh when opening the file in batch mode
Mesh 3;
