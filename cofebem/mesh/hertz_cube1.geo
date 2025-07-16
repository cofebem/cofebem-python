SetFactory("OpenCASCADE");

// ---- geometric parameters ----
L  = 40;          // length  (x)
W  = 40;          // width   (y)
H  = 20;          // height  (z)

// ---- desired mesh sizes ----
hMin = 0.20;      // finest size at focal point
hMax = 5.00;      // coarsest size away from it
rMin = 2;         // sphere of influence (full hMin)
rMax = 10;        // distance where size reaches hMax

// ---- create the box ----
Box(1) = {0, 0, 0,  L, W, H};   // bottom-left corner at (0,0,0)

// ---- point that drives the local refinement ----
pRef = newp;
Point(pRef) = {L/2, W/2, H, hMin};

// ---- size-field definition (Attractor + Threshold) ----
Field[1] = Attractor;
Field[1].PointsList = {pRef};          

Field[2] = Threshold;
Field[2].IField   = 1;                  
Field[2].LcMin    = hMin;
Field[2].LcMax    = hMax;
Field[2].DistMin  = rMin;
Field[2].DistMax  = rMax;

Background Field = 2;                  

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
Print "hertz_cube.vtk";