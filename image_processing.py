import cv2
import os


billede_mappe = "King Domino dataset/Cropped and perspective corrected boards"
output_mappe = "output"  #Den her mappe gemmer de "nye billeder"


if not os.path.exists(output_mappe):
    os.makedirs(output_mappe)


for billede_fil in os.listdir(billede_mappe):
    billede_sti = os.path.join(billede_mappe, billede_fil)

    
    billede = cv2.imread(billede_sti)
    if billede is None:
        print(f"øv kunne ikke læse billedet: {billede_fil}")
        continue
    
    
    billede_grå = cv2.cvtColor(billede, cv2.COLOR_BGR2GRAY) #Her BLIVER BILLEDER LAVET TIL GRÅSKALA
    billede_kanter = cv2.Canny(billede_grå, 100, 200) #Den hER FINDER Kanter lol, så billeder får tydelige kanter, ved ikke om man skal det

    
    output_sti = os.path.join(output_mappe, billede_fil) #Billederne kommer på solo ahah, ej output mappe, 
    cv2.imwrite(output_sti, billede_kanter)

    print(f"Behandlet og gemt: {billede_fil}")
