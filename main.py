import math
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy import signal
from Libraries.zplane import zplane
import scipy.linalg as la


def loadImage(filename):
    img = np.load("Images/In/" + filename)
    
    img_x = int(len(img[0]))
    img_y = int(len(img))
    imgFiltree = np.zeros((img_y, img_x))
    
    return img, imgFiltree


def filtrageAberrations(img, imgFiltree):
    # fonction de transfert :
    b = np.poly([0.9 * np.exp(1j * np.pi / 2), 0.9 * np.exp(-1j * np.pi / 2), 0.95 * np.exp(1j * np.pi / 8),
                 0.95 * np.exp(-1j * np.pi / 8)])  # Zeros
    a = np.poly([0, -0.99, -0.99, 0.88])  # Poles
    
    # Poles et zeros
    # zplane(b, a)
    
    # Poles et zeros de la fonction de transfert inverse
    # zplane(a, b)
    # La fonction de transfert inverse est stable car tous les POLES sont a l'interieur du cercle, donc
    # possede un module < 1
    
    for i in range(0, len(img[0]) - 1):
        imgFiltree[i] = signal.lfilter(a, b, img[i])
    
    plt.figure()
    plt.title("Image originale avec aberrations")
    plt.imshow(img)
    
    matplotlib.pyplot.gray()
    
    plt.figure()
    plt.title("Image filtree par la fonction de transfert inverse")
    plt.imshow(imgFiltree)
    
    return imgFiltree


# def rotationImage():
# # Transformation lineaire de l'image (90 degree vers la droite)
# # Appliquer une matrice de rotation pour identifier la nouvelle position de chacun des pixels
# # Egalement interprete comme un changement de base dans le plan qui constitue un espace vectoriel 2D
# # *** L'origine de l'image est positionnee en haut de l'image a gauche plutot qu'en bas a gauche
# # Il faut donc faire une correction avant de pouvoir appliquer la matrice de rotation qui suppose un systeme cartesion
#

# *** fe = pixel/metre
# Conception d'un filtre passe-bas RII par 2 methodes :
def filtrageBruit(img, imageFiltree, methode, type):
    fe = 1600
    
    # 1 : Appliquer la methode de transformation bilineaire pour obtenir les coefficients du filtre a partir de
    # la fct de transfert H(s) d'un filtre analogique connu
    # convertir un filtre analogique passe-bas Butterworth d'ordre 2 en un filtre numerique
    # Hs = 1 / (((s/wc)^2) + (math.sqrt(2) * (s/wc)) + 1)
    if methode == 1:
        fc = 500
        # 2: Utiliser les fonctions de Python
    
    if methode == 2:
        N = 0
        a = 0
        b = 0
        
        if type == "Butterworth":
            # Filtre Butterworth (N=5)
            # wp=Passband, ws=Stopband, gpass=max loss in passband (dB), gstop=min attenuation in stopband (dB)
            N, Wn = signal.buttord(500, 750, 0.2, 60, fs=fe)
            b, a = signal.butter(N, Wn, 'lowpass', False, output='ba', fs=fe)
        
        if type == "Cheby1":
            # Filtre Chebyshev type I (N=4)
            # wp=Passband, ws=Stopband, gpass=max loss in passband (dB), gstop=min attenuation in stopband (dB)
            N, Wn = signal.cheb1ord(500, 750, 0.2, 60, fs=fe)
            # N=Order, rp=max ripple below unity gain in passband (dB), Wn, btype=type
            b, a = signal.cheby1(N, 1, Wn, 'lowpass', False, output='ba', fs=fe)
        
        if type == "Cheby2":
            # Filtre Chebyshev type II (N=4)
            # wp=Passband, ws=Stopband, gpass=max loss in passband (dB), gstop=min attenuation in stopband (dB)
            N, Wn = signal.cheb2ord(500, 750, 0.2, 60, fs=fe)
            # N=Order, rs=min att required in stopband (dB), Wn, btype=type
            b, a = signal.cheby2(N, 60, Wn, output='ba', fs=fe)
        
        if type == "Elliptique":
            # Filtre elliptique (N=3)
            # wp=Passband, ws=Stopband, gpass=max loss in passband (dB), gstop=min attenuation in stopband (dB)
            N, Wn = signal.ellipord(500, 750, 0.2, 60, fs=fe)
            # N=Order, rp=max ripple below unity gain in passband (dB), rs=min att required in stopband (dB), Wn, btype=type
            b, a = signal.ellip(N, 1, 60, Wn, output='ba', fs=fe)
        
        # Poles et zeroes
        # zplane(b, a)
        
        print(N)
        
        for i in range(0, len(img[0]) - 1):
            imageFiltree[i] = signal.lfilter(b, a, img[i])
    
    matplotlib.pyplot.gray()

    plt.figure()
    plt.title("Image bruitee originale")
    plt.imshow(img)

    plt.figure()
    plt.title("Image filtree par le filtre passe-bas RII choisi")
    plt.imshow(imgFiltree)
    
    return imgFiltree


def compressionImage(image):
    # *** Principle Component Analysis (PCA), determiner une base orthogonale ou le premier element permet d'extraire le max d'informations,
    # le 2ieme un peu moins et ainsi de suite. On pourra laisser tomber les elements de la base qui contiennent le moins d'info
    
    # 1: Calcul de la matrice de covariance de l'image
    # *** Chaque colonne de l'image est un vecteur de N-dimension ou N est le nombre de pixels dans une colonne.
    # Utiliser la fonction python numpy.cov()
    imgCov = numpy.cov(image)
    
    # 2: Determiner les vecteurs propres, qui forment une base de vecteurs independants
    eigvals, eigvecs = la.eig(imgCov)
    
    # 3: Construire une matrice de passage pour exprimer l'image selon cette nouvelle base
    # Les lignes de la matrice de passage permettant de passer de la base originale vers cette nouvelle base seront composées des vecteurs
    # propres de la matrice de covariance, c’est-à-dire chaque ligne de la matrice de passage sera un vecteur propre différent
    
    # 4: Fixer a zero un certain nombre de lignes (respectivement 50% et 70% des lignes)
    
    # 5: Appliquer la matrice de passage inverse pour revenir a l'image originale (Voir note 8 du guide etudiant p.9 du pdf)
    # (autre fonction maybe ?)


if __name__ == '__main__':
    # Filtrage des aberrations en appliquant un filtre numérique
    img, imgFiltree = loadImage("goldhill_aberrations.npy")
    imgAberrationsFiltree = filtrageAberrations(img, imgFiltree)
    
    # # Rotation de l'image
    # rotationImage()
    
    # # Elimination du bruit en haute frequence
    # # choix du type de filtre = Butterworth/Cheby1/Cheby2/Elliptique
    img, imgFiltree = loadImage("goldhill_bruit.npy")
    imgBruitFiltree = filtrageBruit(img, imgFiltree, 2, type="Elliptique")
    
    # # Compression de l'image
    compressionImage(imgFiltree)
    
    plt.show()
    
    exit(1)
