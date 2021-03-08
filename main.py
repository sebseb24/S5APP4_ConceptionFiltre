import math
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy import signal
import zplane
import scipy.linalg as la
import statistics


def loadImage(filename):
    return np.load("Images/In/" + filename)


def filtrageAberrations(img):
    img_x = int(len(img[0]))
    img_y = int(len(img))
    imgFiltree = np.zeros((img_y, img_x))

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


def rotationImage(img, type="npy"):
    img_x = int(len(img[0]))
    img_y = int(len(img))

    if type == "png":
        imageFiltree = np.zeros((img_y, img_x, 4))

    else:
        imageFiltree = np.zeros((img_y, img_x))

    theta = np.radians(-90)
    matriceRotation = np.array(((np.cos(theta), -np.sin(theta)),
                                (np.sin(theta), np.cos(theta))))

    for x in range(0, len(img[0]) - 1):
        for y in range(0, len(img[0]) - 1):
            v = np.array((x, y))
            new_v = matriceRotation.dot(v)
            new_v[1] += (img_x - 1)

            nx = int(new_v[0])
            ny = int(new_v[1])

            imageFiltree[nx][ny] = img[x][y]

    plt.figure()
    plt.title("Image de coté")
    plt.imshow(img)

    plt.figure()
    plt.title("Image après matrice de rotation")
    plt.imshow(imageFiltree)

    return imageFiltree


# *** fe = pixel/metre
# Conception d'un filtre passe-bas RII par 2 methodes :
def filtrageBruit(img, methode, type):
    img_x = int(len(img[0]))
    img_y = int(len(img))
    imageFiltree = np.zeros((img_y, img_x))

    fe = 1600

    # convertir un filtre analogique passe-bas Butterworth d'ordre 2 en un filtre numerique
    # Hs = 1 / (((s/wc)^2) + (math.sqrt(2) * (s/wc)) + 1)
    if methode == 1:

        # fonction de transfert :
        b = np.poly([-1, -1])  # Zeros
        a = np.poly([-0.519634, 0.224966])  # Poles

        # Poles et zeros
        # zplane(b, a)

        # Poles et zeros de la fonction de transfert inverse
        # zplane(a, b)
        # La fonction de transfert inverse est stable car tous les POLES sont a l'interieur du cercle, donc
        # possede un module < 1

        for i in range(0, len(img[0]) - 1):
            imageFiltree[i] = signal.lfilter(b, a, img[i])

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

        print("Type de filtre choisi : " + type)
        print("Ordre du filtre : " + str(N))

        for i in range(0, len(img[0]) - 1):
            imageFiltree[i] = signal.lfilter(b, a, img[i])

    matplotlib.pyplot.gray()

    plt.figure()
    plt.title("Image bruitee originale")
    plt.imshow(img)

    plt.figure()
    plt.title("Image filtree par le filtre passe-bas RII choisi")
    plt.imshow(imageFiltree)

    return imageFiltree


def compressionImage(img, compressValue):
    # *** Principle Component Analysis (PCA), determiner une base orthogonale ou le premier element permet d'extraire le max d'informations,
    # le 2ieme un peu moins et ainsi de suite. On pourra laisser tomber les elements de la base qui contiennent le moins d'info

    matCov = np.cov(img)  # matrice covariance image original

    eigvals, eigvecs = la.eig(matCov)  # valeur propres et vecteurs propres de la matrice covariance

    img_Vec = np.matmul(img,
                        eigvecs)  # image de base vecteur propres, obtenu par la multiplication des vecteurs propres et de l'image de base original
    eigvecs_inv = la.inv(eigvecs)  # calcule vecteur propres inverse

    # boucle qui vérifier le modulo 2 pour la compression a 50% qui en résulte d'une élimination d'une ligne sur 2
    for i in range(0, len(img_Vec)):
        rep = i % 2
        if rep == 0:
            img_Vec[i] = 0
    # boucle qui permet d'enlever une image sur 5 (selon modulo 5) ajouté au traitement du modulo 2, permet de passez de 50 a 70 % de compression [(1/2) + (1/5) = 0.70]
    if compressValue == 70:
        for i in range(0, len(img_Vec)):
            rep = i % 5
            if rep == 0:
                img_Vec[i] = 0

    img_Comp = np.matmul(img_Vec, eigvecs_inv)  # retour de l'image en base original le traitement

    # affichage de l'image compressé
    plt.figure()
    plt.gray()
    plt.imshow(img_Comp)
    plt.title('Image compressée à %i pourcent' % compressValue)


if __name__ == '__main__':
    # Choix entre mode image complete ou images separees
    image_complete = True

    if image_complete:
        img = loadImage("image_complete.npy")

        # Filtrage des aberrations en appliquant un filtre numérique
        imgAberrationsFiltree = filtrageAberrations(img)

        # # Rotation de l'image
        imgTournee = rotationImage(imgAberrationsFiltree, "npy")

        # # Elimination du bruit en haute frequence
        # # choix du type de filtre = Butterworth/Cheby1/Cheby2/Elliptique
        imageFinale = filtrageBruit(imgTournee, 2, type="Elliptique")

    else:
        # Filtrage des aberrations en appliquant un filtre numérique
        img = loadImage("goldhill_aberrations.npy")
        imgAberrationsFiltree = filtrageAberrations(img)

        # # Rotation de l'image
        img = matplotlib.image.imread("Images/In/goldhill_rotate.png")
        imgTournee = rotationImage(img, "png")

        # # Elimination du bruit en haute frequence
        # # choix du type de filtre = Butterworth/Cheby1/Cheby2/Elliptique
        img = loadImage("goldhill_bruit.npy")
        imageFinale = filtrageBruit(img, 2, type="Elliptique")

    # Compression de l'image
    # mettre 50 pour une compression de 50% ou 70 pour une compression de 70%
    compressionImage(imageFinale, 70)

    plt.show()

    exit(1)
