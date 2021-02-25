import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Libraries.zplane import zplane



def P1():
    b = np.poly([0.8*1j, -0.8*1j])
    a = np.poly([0.95*np.exp(1j*np.pi/8),0.95*np.exp(-1j*np.pi/8)])

    #a
    zplane(b, a)

    #b filtre stable car pole sur 1

    #c
    # w = La frequence a laquelle h a ete computed. par defaut, w est normalisee entre 0 et pi rad/ech
    # h = La reponse en frequence du filtre en nombre complexe
    w, h = signal.freqz(b, a)

    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')

    plt.savefig("b")

    #d
    tabZero = np.zeros(501)
    tabZero[251] = 1
    # Filtrage de l'impulsion avec le filtre pour obtenir h[n] (reponse impulsionnelle?)
    signalHn = signal.lfilter(b,a, tabZero)
    # Calcul de la FFT
    signalfft = np.fft.fft(signalHn)

    plt.figure()
    plt.plot(signalHn)

    plt.figure()
    plt.plot(20*np.log10(abs(signalfft))[:250])

    #e
    # Filtre avec le filtre inverse (numerateur et denom inverse)
    sinalHnInv = signal.lfilter(a,b,signalHn)
    plt.figure()
    plt.plot(sinalHnInv)

    plt.show()

def P2():
    a = np.poly([np.exp(1j*np.pi/16-0.05), np.exp(-1j*np.pi/16-0.05)])
    b = np.poly([np.exp(1j*np.pi/16), np.exp(-1j*np.pi/16)])

    zplane(b, a)

    # Reponse en frequence
    w, h = signal.freqz(b, a)

    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')

    plt.show()


def P3():
   N, Wn = signal.buttord(2500,3500, 0.2, 40, fs=48000)
   b, a = signal.butter(N, Wn, 'lowpass', False, output='ba', fs=48000)

   zplane(b,a)

   print(N)

   w, h = signal.freqz(b, a)

   fig, ax1 = plt.subplots()
   ax1.set_title('Digital filter frequency response')
   ax1.plot(w, 20 * np.log10(abs(h)), 'b')
   ax1.set_ylabel('Amplitude [dB]', color='b')
   ax1.set_xlabel('Frequency [rad/sample]')

   ax2 = ax1.twinx()
   angles = np.unwrap(np.angle(h))
   ax2.plot(w, angles, 'g')
   ax2.set_ylabel('Angle (radians)', color='g')
   ax2.grid()
   ax2.axis('tight')

   plt.show()

    #cheb 1  N = 8
    #cheb 2  N = 8
    #ellip  N = 5

def P4():

    #a   T(x,y)=(2x, y/2)

    img = plt.imread('goldhill.png')

    img_x = int(2*len(img[0]))
    img_y = int(len(img)/2)
    imgT = np.zeros((img_y,img_x))

    for y in range(0, len(img[0])):
        for x in range(0, len(img[0])):
            if y%2==0:
                imgT[int(y/2)][2*x] = img[y][x]


    plt.gray()

    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(imgT)

    plt.show()

if __name__ == '__main__':
    P3()
    exit(1)
