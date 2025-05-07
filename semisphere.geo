// hemisphere.geo
// Ce fichier crée une hémisphère (moitié supérieure d'une sphère de rayon R)
// en utilisant une différence booléenne entre une sphère et une boîte qui élimine la partie inférieure.

SetFactory("OpenCASCADE");

// Paramètre : rayon de la sphère
R = 1.0;

// 1. Créer une sphère complète de rayon R centrée en (0,0,0)
Sphere(1) = {0, 0, 0, R};

// 2. Créer une boîte couvrant la moitié inférieure de la sphère.
//    La boîte s'étend de z = -R à z = 0 et couvre entièrement la sphère en x et y.
Box(2) = {-R, -R, -R, 2*R, 2*R, R};

// 3. Soustraire la boîte de la sphère pour éliminer la partie avec z < 0,
//    ce qui laisse l'hémisphère supérieur.
hemisphere[] = BooleanDifference{ Volume{1}; }{ Volume{2}; };

// 4. Synchroniser le modèle afin que l'opération booléenne soit prise en compte.
Synchronize();

// 5. Augmenter la densité du maillage en réduisant la taille caractéristique.
//    Plus ces valeurs sont petites, plus le maillage sera fin (et donc, le nombre de nœuds sera plus élevé).
Mesh.CharacteristicLengthMin = 0.1;
Mesh.CharacteristicLengthMax = 0.1;

// 6. Définir un groupe physique pour l'hémisphère, pour pouvoir le référencer dans FEniCS.
Physical Volume("Hemisphere") = { hemisphere[0] };

