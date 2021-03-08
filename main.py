# #################################################################
# Auteurs : Sébastien Pomerleau et Gabriel Roy                    #
# S5 APP 4 Informatique : Conception d'un filtre et compression   #
# Date : 9 mars 2021                                              #
# #################################################################

from Filtres import filtrage_image

if __name__ == '__main__':
    # Choix entre mode image complète ou images separées
    image_complete = True

    # Choix de la méthode à utiliser pour la conception du filtre passe-bas
    # 1 : Approche par transformation bilinéaire
    # 2 : Fonction Python
    choix_methode_filtre_pb = 1

    # choix du type de filtre = Butterworth/Cheby1/Cheby2/Elliptique
    type_filtre = "Elliptique"
    
    # Choix du pourcentage de compression (50/70)
    taux_compression = 70
    
    # Affichage des zplane
    show_zplane = False
    
    filtrage_image(image_complete, choix_methode_filtre_pb, type_filtre, taux_compression, show_zplane)

    exit(1)
