import math

def filtrageAberrations():
    # fonction de transfert :
        # Hz = ((z-0.9*math.exp((1j*math.pi)/2)) * (z-0.9*math.exp((-1j*math.pi)/2)) * (z-0.95*math.exp((1j*math.pi)/8)) * (z-0.95*math.exp((-1j*math.pi)/8))) \
        #         / (z*((z+0.99)^2)*(z-0.8))

    # Appliquer l'inverse de la fonction de transfert a l'image
    # ** Appliquer le filtre selon les colonnes de l'image uniquement, donc considerez chaque colonne
    # comme un signal different pour appliquer le filtre successivement sur chacune des colonnes
    # Conversion en ton de gris
    # matplotlib.pyplot.gray()
    # img_color = matplotlib.image.imread('imagecouleur.png')
    # img = numpy.mean(img_color, -1)
    # Veuillez noter que pour lire une matrice dans le format de python .npy, il suffit de faire, p.ex.:
    # img_load = np.load("goldhill_aberrations.npy")

    # Verifier si la fonction de transfert inverse est stable (plot poles et zeros ?)

def rotationImage():
    # Transformation lineaire de l'image (90 degree vers la droite)
        # Appliquer une matrice de rotation pour identifier la nouvelle position de chacun des pixels
        # Egalement interprete comme un changement de base dans le plan qui constitue un espace vectoriel 2D
        # *** L'origine de l'image est positionnee en haut de l'image a gauche plutot qu'en bas a gauche
        # Il faut donc faire une correction avant de pouvoir appliquer la matrice de rotation qui suppose un systeme cartesion

def filtrageBruit():
    # *** fe = pixel/metre
    # Conception d'un filtre passe-bas RII par 2 methodes :
    # 1 : Appliquer la methode de transformation bilineaire pour obtenir les coefficients du filtre a partir de
    # la fct de transfert H(s) d'un filtre analogique connu
    # convertir un filtre analogique passe-bas Butterworth d'ordre 2 en un filtre numerique
    # Hs = 1 / (((s/wc)^2) + (math.sqrt(2) * (s/wc)) + 1)
    fc = 500
    fe = 1600

    # 2: Utiliser les fonctions de Python
    # Il faut respecter les tolerances suivantes :
        # gain de 0dB +- 0.2dB sur la majeure partie de la bande passante 0 a 500 Hz
        # gain doit etre au moins -60dB pour les frequences plus elevees que 750 Hz
        # Choisir le type de filtre ayant le plus petit ordre possible (Butterworth, Chebyshev type I, Chebyshev type II, elliptique)


def compressionImage():
    # *** Principle Component Analysis (PCA), determiner une base orthogonale ou le premier element permet d'extraire le max d'informations,
    # le 2ieme un peu moins et ainsi de suite. On pourra laisser tomber les elements de la base qui contiennent le moins d'info

    # 1: Calcul de la matrice de covariance de l'image
    # *** Chaque colonne de l'image est un vecteur de N-dimension ou N est le nombre de pixels dans une colonne.
    # Utiliser la fonction python numpy.cov()

    # 2: Determiner les vecteurs propres, qui forment une base de vecteurs independants

    # 3: Construire une matrice de passage pour exprimer l'image selon cette nouvelle base
    # Les lignes de la matrice de passage permettant de passer de la base originale vers cette nouvelle base seront composées des vecteurs
    # propres de la matrice de covariance, c’est-à-dire chaque ligne de la matrice de passage sera un vecteur propre différent

    # 4: Fixer a zero un certain nombre de lignes (respectivement 50% et 70% des lignes)

    # 5: Appliquer la matrice de passage inverse pour revenir a l'image originale (Voir note 8 du guide etudiant p.9 du pdf)
    # (autre fonction maybe ?)


if __name__ == '__main__':
    # Filtrage des aberrations en appliquant un filtre numérique
    filtrageAberrations()

    # Rotation de l'image
    rotationImage()

    # Elimination du bruit en haute frequence
    filtrageBruit()

    # Compression de l'image
    compressionImage()

    exit(1)