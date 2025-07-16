# Fast Boundary Element Method for Finite-Geometry Problems in Contact Mechanics

**Auteur**: Yahya Boye  
**Encadrants**: Stéphanie Chaillat (ENSTA – IP Paris), Vladislav Yastrebov (Mines Paris – PSL), Jérémy Bleyer (ENPC, IP Paris)
**Date**: 22 mai 2025  

---

## 1. Contexte et Motivation

### 1.1 Définition de la mécanique du contact
- Interaction entre deux corps qui se touchent.
- Implique la non-pénétration, la transmission de forces, le frottement, l'usure.

### 1.2 Domaines d'application
- Interface roue/rail, engrenages, systèmes de sécurité, formage des métaux.
- Biomécanique, sismologie (failles tectoniques), etc.

### 1.3 Défis spécifiques
- Couplages multiphysiques (thermique, fluide, mécanique).
- Phénomènes multi-échelles, évolution des surfaces, fissures, 3rd body problem.
- Modélisation robuste nécessaire pour les géométries complexes.

---

## 2. Méthodes Numériques Classiques

### 2.1 Modèles analytiques ### a jeter si long
- Contact hertzien : solution fermée pour deux solides élastiques sans frottement.
- Solution de Boussinesq : déplacement sous une charge ponctuelle normale.

### 2.2 Formulation forte
- Équilibre mécanique sur domaines Ω₁ et Ω₂.
- Conditions de Dirichlet, Neumann, et surface de contact potentielle Γc.

### 2.3 Conditions de contact
- Formulation de Signorini (sans frottement) : g ≥ 0, σₙ ≤ 0, gσₙ = 0.
- Complémentarité : aucune pénétration ni adhésion, pas de cisaillement.

### 2.4 Méthode de pénalisation
- Ajout d’un ressort fictif de raideur ε.
- Problème approché par régularisation du potentiel total.

### 2.5 Méthode des multiplicateurs de Lagrange
- Problème en point selle avec inconnues supplémentaires λ.
- Formulation mixte : conditions sur u et λ.

### 2.6 Méthode de Lagrangien augmenté
- Combinaison de pénalisation et multiplicateurs.
- Algorithme itératif : mise à jour alternée de u et λ.

### 2.7 Limites des approches classiques
- Pénalisation : mauvaise conditionnement pour ε grand, ajustements nécessaires.
- Multiplicateurs : système indéfini plus grand, stabilité inf-sup.
- Lagrangien augmenté : coût supplémentaire, paramétrage sensible.
**Methode robuste et non intrusive et qui peut etre accelerer avec les H-matrices**

---

## 3. Notre Approche : Méthode CoFeBem

### 3.1 Vue d’ensemble du solveur
- Formulation du problème de contact comme un problème auxiliaire.
- Séparation entre la simulation (FEM/BEM) et le solveur de contact.

### 3.2 Algorithme proposé
1. Discrétisation (FEM/BEM), construction des opérateurs.
2. Construction de la matrice de compliance de contact Sc (Par FEM ou BEM)
3. Partitionnement via arbre binaire → H-matrice.
4. Approximation des blocs lointains par décompositions faible rang.
5. Formulation LCP : g = Sc λ + g₀.
6. Résolution : CCG, Lemke, NNLS...
7. Application des forces de contact sur le système global.

### 3.3 Avantages
- Indépendance vis-à-vis du solveur principal.
- Réutilisabilité de Sc.
- Accélération via matrices hiérarchiques (H-matrices).

### 3.4 Construction de Sc
- **Par complément de Schur** :
  - Condensation des degrés de liberté intérieurs.
  - Interprétation : Sc = compliance condensée sur l’interface.
- **Par échantillonnage direct** :
  - Simulations élémentaires pour estimer les réponses locales des noeuds de la région de contact.
- **Par Collocation BEM** :
  - Sc par la méthode des élèments de frontières.

### 3.5 Accélération par H-matrices
- **Partition des dofs** :
  - PCA Clustering
- **Approximation faible range** :
  - SVD tronquée, ACA, ACA+

---

## 4. Résultats et Discussion

### 4.1 Tests numériques
- **Ironing test** : test de validation du code.
- **Hémisphère** : cas test adaptés

### 4.2 Comparaisons
- **Schur vs Sampling** :
  - Scalabilité O(N²) confirmée.
  - Iterative GMRES moins performant que Schur direct.
- **Optimisation par symétrie** :
  - Réduction du coût pour géométries symétriques, exemple du donut.
- **Solveurs LCP** :
  - **CCG** : mieux adapté aux grands problèmes, solution approchée.
  - **Lemke** : exact mais coûteux en temps et mémoire.

- **Résolution du contact avec Sc H-matrice** :
  - Comparaison complexité Sc Full et Sc H-matrice.
  - Résoudre un problème le plus large possible en HPC.
  - Benchmark avec un logiciel commercial 

---

## 5. Conclusion

### 5.1 Contributions
- Preuve de concept fonctionnelle.
- Adaptabilité aux problèmes axisymétriques.


### 5.2 Forces de l’approche
- **Modularité** : décorrélation entre la simulation et le contact.
- **Scalabilité** : compression H-matrice.
- **Robustesse** : adaptabilité à toutes géométries et matériaux.

---

## 6. Perspectives

### 6.1 Développements prévus
- Implémentation complète de Sc en CBEM et SGBEM.
- Comparaison entre collocation directe et formulation symétrique.

### 6.2 Complexification physique
- Introduction du frottement (bipotentiel de de Saxcé).
- Contact déformable–déformable.
- Matériaux non-linéaires : plasticité, visco-, hyperélasticité.

### 6.3 Performances et interfaçage
- Compression par ACA, troncature SVD.
- Comparaison des solveurs LCP : CG, NNLS, Lemke.
- Interfaçage avec FEniCSx, Zset, MFEM, etc.

---

##### Ar = 0.05 et At = 0.1
