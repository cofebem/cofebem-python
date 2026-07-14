SetFactory("OpenCASCADE");

// ---------------------------
// Parameters
// ---------------------------
R  = 1.0;       // radius of the sphere
lc = R / 8;     // target mesh size

// ---------------------------
// Mesh options
// ---------------------------
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.Algorithm3D             = 4;    // 3D Delaunay
Mesh.ElementOrder            = 1;    // P1 tets
Mesh.MshFileVersion          = 2.2;  

// ---------------------------
// Geometry: full sphere
// ---------------------------
Sphere(1) = {0, 0, 0, R};

// Box that contains only the upper half-space (z >= 0)
eps = R * 1e-3;
Box(2) = {-R - eps, -R - eps, 0, 2*R + 2*eps, 2*R + 2*eps, R + eps};

// Intersection = hemisphere (solid volume)
hem[] = BooleanIntersection{ Volume{1}; Delete; }{ Volume{2}; Delete; };

// Hemisphere volume tag
hemVol = hem[0];

// All boundary surfaces of the hemisphere
allSurf[] = Surface In Volume{hemVol};

// Flat circular base at z = 0
baseSurf[] = Surface In BoundingBox(-R - eps, -R - eps, -eps,
                                   R + eps,  R + eps,  eps);

// Curved spherical surface (remaining surfaces)
curvedSurf[] = {}; 
For i In {0:#allSurf[]-1}
  If (!InList(baseSurf[], allSurf[i]))
    curvedSurf[] += {allSurf[i]};
  EndIf
EndFor

// Physical groups 
Physical Volume("Hemisphere")         = {hemVol};
Physical Surface("Hemisphere_Base")   = {baseSurf[]};
Physical Surface("Hemisphere_Shell")  = {curvedSurf[]};
