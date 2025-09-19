SetFactory("OpenCASCADE");

// ---------------- Parameters ----------------
R  = 1.0;          // radius
lc = 0.10;         // target mesh size on the surface 0.15
xc = 0; yc = 0; zc = 0; // center

// Create a solid sphere (a volume); we'll mesh only its boundary (surface)
v1 = newv;
Sphere(v1) = {xc, yc, zc, R};

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

Mesh.RecombineAll = 0;

Mesh.Algorithm = 6; // Frontal-Delaunay for surfaces

s[] = Boundary{ Volume{v1}; };
Physical Surface("SphereSurface") = {s[]};
// (Optional) keep the volume as a physical group if you ever do 3D:
Physical Volume("SphereVolume") = {v1};
