SetFactory("OpenCASCADE");

// ---- geometric parameters ----
L  = 40;          // length  (x)
W  = 40;          // width   (y)
H  = 20;          // height  (z)
R  = 5;

// ---- desired mesh sizes ----
hMin = 0.20;      // finest size at focal point
hMax = 5.00;      // coarsest size away from it
rMin = 2;         // sphere of influence (full hMin)
rMax = 10;        // distance where size reaches hMax

// ---- create the box ----
Box(1) = {0, 0, 0,  L, W, H};   // bottom-left corner at (0,0,0)

Point(9) = {L/2, W/2, H, hMin};

// ---- surface recombination (quads) ----
Mesh.RecombineAll = 1;                  

// ---- hexahedral meshing options ----
Mesh.Algorithm       = 6;              
Mesh.Algorithm3D     = 12;            
Mesh.SubdivisionAlgorithm = 1;          
Mesh.CharacteristicLengthMin = hMin;
Mesh.CharacteristicLengthMax = hMax;
Mesh.Optimize = 1;                     

// Final mesh
Mesh 3;
Print "hertz_cube_vy.vtk";
//+
Delete {
  Volume{1};
  Surface{6}; 
}
//+
Sphere(1) = {L/2., W/2., H, R, -Pi/4, Pi/4, 2*Pi};
