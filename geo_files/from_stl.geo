SetFactory("OpenCASCADE");

// 1 Import STL
Merge "Pikachu.stl";

// 2 Reconstruct CAD-like patches from triangulation
angle = 40 * Pi/180;
ClassifySurfaces(angle, 1, 1, 180 * Pi/180);
CreateGeometry;

// (Optional but helpful)
Geometry.OCCSewFaces = 1;
Coherence;

// 3 Build a volume from ALL surfaces found
s[] = Surface{:};
Surface Loop(1) = {s[]};
Volume(1) = {1};

// 4 Physical groups
Physical Volume("Pikachu") = {1};
Physical Surface("Boundary") = {s[]};