SetFactory("OpenCASCADE");

// =================== Parameters ===================
L  = 1.0;            // cube edge
order = 1;           // 1 or 2
nZ = 8;              // hex layers through thickness (z)

// Angular divisions per quadrant for every circular arc
nThetaQ = 12;

// Radial divisions:
//  - nRadInner: center -> first circle (fan)
//  - nRadRing:  between consecutive circles
//  - nRadOuter: last circle -> mid-edge
nRadInner = 2;
nRadRing  = 2;
nRadOuter = 2;

// Concentric circles (radii from center); last must be < L/2
nr = 3;                        // number of circles (>=1)
rMin = 0.12*L;                 // inner circle radius
rMax = 0.45*L;                 // outer circle radius (< L/2)
If (nr < 1)
  nr = 1;
EndIf

// Corner rectangles edge divisions (half-edges)
nEdgeCorner = 12;

// Mesh options
Mesh.ElementOrder = order;
Mesh.Optimize = 1;
Mesh.HighOrderOptimize = 1;
Mesh.RecombineAll = 1;
Mesh.MshFileVersion = 2.2;
Mesh.SaveAll = 1;

// =================== Coordinates ===================
cx = 0.5*L; cy = 0.5*L; cz = L;

// =================== Top perimeter points ===================
Point(1) = {0, 0, cz};
Point(2) = {L, 0, cz};
Point(3) = {L, L, cz};
Point(4) = {0, L, cz};

Point(5) = {cx, 0,  cz};   // bottom-mid (y=0)
Point(6) = {L,  cy, cz};   // right-mid  (x=L)
Point(7) = {cx, L,  cz};   // top-mid    (y=L)
Point(8) = {0,  cy, cz};   // left-mid   (x=0)

Point(20) = {cx, cy, cz};  // center

// Mid-edge connectors (between midpoints)
Line(111) = {6, 7};  // right-mid -> top-mid    (around corner 3)
Line(112) = {7, 8};  // top-mid   -> left-mid   (around corner 4)
Line(113) = {8, 5};  // left-mid  -> bottom-mid (around corner 1)
Line(114) = {5, 6};  // bottom-mid-> right-mid  (around corner 2)

// Corner half-edges (for rectangular patches)
Line(131) = {2, 6};
Line(132) = {6, 3};
Line(133) = {3, 7};
Line(134) = {7, 4};
Line(135) = {4, 8};
Line(136) = {8, 1};
Line(137) = {1, 5};
Line(138) = {5, 2};

// =================== Concentric circles (points + arcs) ===================
r[] = {};
For k In {1:nr}
  r[k] = rMin + (rMax - rMin)*(k - 1)/(nr - 1 + 1e-12); // linear spacing
EndFor

// IDs will be generated programmatically and stored in fields
// Points at each radius along axes
pR[] = {}; pT[] = {}; pL[] = {}; pB[] = {};
arcQ1[] = {}; arcQ2[] = {}; arcQ3[] = {}; arcQ4[] = {};

// Axis radial segments
lCR[] = {}; lCT[] = {}; lCL[] = {}; lCB[] = {};   // center -> first circle
lR[]  = {}; lT[]  = {}; lL[]  = {}; lB[]  = {};   // between successive circles
// Midpoint -> last circle
lRmid = 0; lTmid = 0; lLmid = 0; lBmid = 0;

// --- make points on circles
For k In {1:nr}
  pR[k] = newp; Point(pR[k]) = {cx + r[k], cy, cz};
  pT[k] = newp; Point(pT[k]) = {cx, cy + r[k], cz};
  pL[k] = newp; Point(pL[k]) = {cx - r[k], cy, cz};
  pB[k] = newp; Point(pB[k]) = {cx, cy - r[k], cz};

  // quarter circle arcs (right->top, top->left, left->bottom, bottom->right)
  arcQ1[k] = newl; Circle(arcQ1[k]) = {pR[k], 20, pT[k]};
  arcQ2[k] = newl; Circle(arcQ2[k]) = {pT[k], 20, pL[k]};
  arcQ3[k] = newl; Circle(arcQ3[k]) = {pL[k], 20, pB[k]};
  arcQ4[k] = newl; Circle(arcQ4[k]) = {pB[k], 20, pR[k]};

  If (k == 1)
    // center -> first circle (four axes)
    lCR[k] = newl; Line(lCR[k]) = {20, pR[k]};
    lCT[k] = newl; Line(lCT[k]) = {20, pT[k]};
    lCL[k] = newl; Line(lCL[k]) = {20, pL[k]};
    lCB[k] = newl; Line(lCB[k]) = {20, pB[k]};
  Else
    // between consecutive circles (four axes)
    lR[k] = newl; Line(lR[k]) = {pR[k-1], pR[k]};
    lT[k] = newl; Line(lT[k]) = {pT[k-1], pT[k]};
    lL[k] = newl; Line(lL[k]) = {pL[k-1], pL[k]};
    lB[k] = newl; Line(lB[k]) = {pB[k-1], pB[k]};
  EndIf
EndFor

// Midpoint -> last circle along axes (closes outer ring sectors)
lRmid = newl; Line(lRmid) = {6, pR[nr]};
lTmid = newl; Line(lTmid) = {7, pT[nr]};
lLmid = newl; Line(lLmid) = {8, pL[nr]};
lBmid = newl; Line(lBmid) = {5, pB[nr]};

// =================== Top 2D patches ===================
// --- Innermost “fan” (center to first circle) — 4 sectors (tri surfaces, recombined)
sQ1_fan = newreg; Curve Loop(sQ1_fan) = { lCT[1], -arcQ1[1], -lCR[1] };
S301 = news; Plane Surface(S301) = {sQ1_fan};

sQ2_fan = newreg; Curve Loop(sQ2_fan) = { lCL[1], -arcQ2[1], -lCT[1] };
S302 = news; Plane Surface(S302) = {sQ2_fan};

sQ3_fan = newreg; Curve Loop(sQ3_fan) = { lCB[1], -arcQ3[1], -lCL[1] };
S303 = news; Plane Surface(S303) = {sQ3_fan};

sQ4_fan = newreg; Curve Loop(sQ4_fan) = { lCR[1], -arcQ4[1], -lCB[1] };
S304 = news; Plane Surface(S304) = {sQ4_fan};

// --- Middle rings between circles (if nr >= 2): 4*(nr-1) quad patches
SRmid[] = {}; // store surface ids for later extrusion
For k In {2:nr}
  // Q1: between arcQ1[k-1] and arcQ1[k]
  cl = newreg; Curve Loop(cl) = { lT[k], -arcQ1[k], -lR[k], arcQ1[k-1] };
  s = news; Plane Surface(s) = {cl}; SRmid[] += {s};

  // Q2
  cl = newreg; Curve Loop(cl) = { lL[k], -arcQ2[k], -lT[k], arcQ2[k-1] };
  s = news; Plane Surface(s) = {cl}; SRmid[] += {s};

  // Q3
  cl = newreg; Curve Loop(cl) = { lB[k], -arcQ3[k], -lL[k], arcQ3[k-1] };
  s = news; Plane Surface(s) = {cl}; SRmid[] += {s};

  // Q4
  cl = newreg; Curve Loop(cl) = { lR[k], -arcQ4[k], -lB[k], arcQ4[k-1] };
  s = news; Plane Surface(s) = {cl}; SRmid[] += {s};
EndFor

// --- Outermost ring: last circle -> mid-edge segments (4 quads)
cl = newreg; Curve Loop(cl) = { lTmid, -arcQ1[nr], -lRmid, -111 };
S311 = news; Plane Surface(S311) = {cl};

cl = newreg; Curve Loop(cl) = { lLmid, -arcQ2[nr], -lTmid, -112 };
S312 = news; Plane Surface(S312) = {cl};

cl = newreg; Curve Loop(cl) = { 113, lBmid, -arcQ3[nr], -lLmid };
S313 = news; Plane Surface(S313) = {cl};

cl = newreg; Curve Loop(cl) = { 114, lRmid, -arcQ4[nr], -lBmid };
S314 = news; Plane Surface(S314) = {cl};

// --- Corner rectangles (4 quads)
cl = newreg; Curve Loop(cl) = {131, 111, -133, 132}; S321 = news; Plane Surface(S321) = {cl};
cl = newreg; Curve Loop(cl) = {134, 112, -135, 133}; S322 = news; Plane Surface(S322) = {cl};
cl = newreg; Curve Loop(cl) = {136, 113, -137, 135}; S323 = news; Plane Surface(S323) = {cl};
cl = newreg; Curve Loop(cl) = {138, 114, -131, 137}; S324 = news; Plane Surface(S324) = {cl};

// =================== Transfinite constraints ===================
// Arcs: same angular divisions for all
Transfinite Curve {arcQ1[], arcQ2[], arcQ3[], arcQ4[]} = nThetaQ+1 Using Progression 1;

// Center fan radial lines
Transfinite Curve {lCR[1], lCT[1], lCL[1], lCB[1]} = nRadInner+1 Using Progression 1;

// Between-circles radial lines
If (nr >= 2)
  // collect all lR/lT/lL/lB from k=2..nr
  auxC[] = {};
  For k In {2:nr}
    auxC[] += { lR[k], lT[k], lL[k], lB[k] };
  EndFor
  Transfinite Curve {auxC[]} = nRadRing+1 Using Progression 1;
EndIf

// Outermost radial lines to midpoints
Transfinite Curve {lRmid, lTmid, lLmid, lBmid} = nRadOuter+1 Using Progression 1;

// Mid-edge segments shared by outer ring and corners
Transfinite Curve {111,112,113,114} = nThetaQ+1 Using Progression 1;

// Corner half-edges
Transfinite Curve {131,132,133,134,135,136,137,138} = nEdgeCorner+1 Using Progression 1;

// Surfaces: transfinite + recombine everywhere
Transfinite Surface {S301,S302,S303,S304}; Recombine Surface {S301,S302,S303,S304};
Transfinite Surface {S311,S312,S313,S314}; Recombine Surface {S311,S312,S313,S314};
Transfinite Surface {S321,S322,S323,S324}; Recombine Surface {S321,S322,S323,S324};
If ( #SRmid[] > 0 )
  Transfinite Surface {SRmid[]};
  Recombine Surface {SRmid[]};
EndIf

// Group all top patches for extrusion
topPatches[] = {S301,S302,S303,S304,S311,S312,S313,S314,S321,S322,S323,S324};
If ( #SRmid[] > 0 )
  topPatches[] += {SRmid[]};
EndIf

// =================== Extrude to 3D hexahedra ===================
out[] = Extrude {0, 0, -L} {
  Surface{topPatches[]};
  Layers{nZ};
  Recombine;
};

// Collect volumes (every input surface yields one volume at out[1], out[5], ...)
allVols[] = {};
// stride is 4 per extruded surface: [top, lateral1, lateral2, volume]
nSurf = #topPatches[];
For i In {0:nSurf-1}
  allVols[] += { out[4*i + 1] };
EndFor

// =================== Physical groups ===================
Physical Volume("Cube") = {allVols[]};

// Side & bottom faces via bounding boxes
eps = 1e-9;
left[]   = Surface In BoundingBox(-eps, 0,       0,        eps,   L,       L);
right[]  = Surface In BoundingBox( L-eps,0,       0,        L+eps, L,       L);
front[]  = Surface In BoundingBox( 0,      -eps,  0,        L,       eps,   L);
back[]   = Surface In BoundingBox( 0,       L-eps,0,        L,       L+eps, L);
bottom[] = Surface In BoundingBox( 0,       0,     -eps,    L,       L,       eps);
top[]    = Surface In BoundingBox( 0,       0,      L-eps,  L,       L,       L+eps);

Physical Surface("Left")   = {left[]};
Physical Surface("Right")  = {right[]};
Physical Surface("Front")  = {front[]};
Physical Surface("Back")   = {back[]};
Physical Surface("Bottom") = {bottom[]};
Physical Surface("Top")    = {top[]};

// Curves transfinite controls (optional visualization)
// Mesh 2; // preview 2D paving first
