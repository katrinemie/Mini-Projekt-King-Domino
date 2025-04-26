import cv2
import numpy as np
import os
import glob
import csv  

class TileClassifier:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
        if not self.image_paths:
           
            raise FileNotFoundError(f" Ingen .jpg billeder fundet i mappen: '{input_folder}'")
        print(f"Fandt {len(self.image_paths)} billeder i '{input_folder}'.") # Feedback

    def classify_tile(self, tile):
       
        if tile is None or tile.size == 0:
            return "Invalid Tile"
        try:
            hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
            
            if hsv.shape[0] > 0 and hsv.shape[1] > 0:
                 
                hue, sat, val = np.mean(hsv.reshape(-1, 3), axis=0)
            else:
                return "Small/Empty Tile"

        except cv2.error as e:
             
            print(f" OpenCV fejl ved behandling af tile: {e}")
            return "Processing Error"

        
        if 21.5 < hue < 27.5 and 225 < sat < 255 and 104 < val < 210:
            return "Field"
        if 25 < hue < 60 and 88 < sat < 247 and 24 < val < 78:
            return "Forest"
        if 90 < hue < 130 and 100 < sat < 255 and 100 < val < 230:
            return "Lake"
        if 34 < hue < 46 and 150 < sat < 255 and 90 < val < 180:
            return "Grassland"
        if 16 < hue < 27 and 66 < sat < 180 and 75 < val < 140:
            return "Swamp"
        if 19 < hue < 27 and 39 < sat < 150 and 29 < val < 80:
            return "Mine"
        
        if sat < 60 and val > 50: 
            return "Structure/Home" 
        return "Unknown"

    def split_to_tiles(self, image):
        h, w = image.shape[:2]
       
        tile_h = max(1, h // 5)
        tile_w = max(1, w // 5)
        
        if h < 5 or w < 5:
             print(f" for lille billede ({w}x{h}), kan ikke opdeles i 5x5 tiles.")
             
             return [[image]]

        tiles = []
        for y in range(5):
            row_tiles = []
            for x in range(5):
                
                y_start = y * h // 5
                y_end = (y + 1) * h // 5
                x_start = x * w // 5
                x_end = (x + 1) * w // 5
                tile = image[y_start:y_end, x_start:x_end]
                row_tiles.append(tile)
            tiles.append(row_tiles)
        return tiles


    def annotate_tiles(self, image, labels):
        annotated = image.copy()
        h, w = image.shape[:2]
        
        tile_h = max(1, h // 5)
        tile_w = max(1, w // 5)
        num_rows = len(labels)
        num_cols = len(labels[0]) if num_rows > 0 else 0

        for row in range(num_rows):
            for col in range(num_cols):
                 
                y_start = row * h // num_rows 
                y_end = (row + 1) * h // num_rows
                x_start = col * w // num_cols
                x_end = (col + 1) * w // num_cols

                label = labels[row][col]
                
                text_pos = (x_start + 5, y_start + 15) 
                cv2.putText(annotated, label, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA) 
                cv2.putText(annotated, label, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA) 
                
                cv2.rectangle(annotated, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
        return annotated

   
    def process_images(self, output_csv_file, show_images=True):
        
        try:
            with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                
                csv_writer = csv.writer(csvfile)

                
                csv_writer.writerow(['filename', 'row', 'column', 'classification'])

                
                for path in self.image_paths:
                    img = cv2.imread(path)
                    if img is None:
                        print(f" Kunne ikke læse billedet: {path}. Springer over.")
                        
                        
                        continue

                    image_filename = os.path.basename(path) 
                    print(f"\n📄 Behandler billede: {image_filename}")

                    tiles = self.split_to_tiles(img)
                    terrain_labels = [] 

                    num_rows_actual = len(tiles)
                    num_cols_actual = len(tiles[0]) if num_rows_actual > 0 else 0

                    
                    for row_idx in range(num_rows_actual):
                        label_row = []
                        for col_idx in range(num_cols_actual):
                            tile = tiles[row_idx][col_idx]
                            label = self.classify_tile(tile)
                            label_row.append(label)

                            
                            csv_writer.writerow([image_filename, row_idx, col_idx, label])

                            

                        terrain_labels.append(label_row)

                    
                    if show_images:
                        if terrain_labels: 
                            annotated = self.annotate_tiles(img, terrain_labels)
                            cv2.imshow("Klassificerede Tiles", annotated)
                            
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                print("Afslutter visning efter brugerinput ('q').")
                                show_images = False 
                                cv2.destroyAllWindows() 
                        else:
                             print(f"  Ingen tiles fundet eller behandlet for {image_filename}, springer annotering over.")


        except IOError as e:
            print(f" Fejl ved skrivning til CSV-fil '{output_csv_file}': {e}")
        except Exception as e:
            print(f" En uventet fejl opstod under billedbehandling eller CSV-skrivning: {e}")
        finally:
            
            if show_images:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    INPUT_FOLDER = 'splitted_dataset/train/cropped' 
    OUTPUT_CSV = 'tile_classifications.csv' #
    SHOW_ANNOTATED_IMAGES = True 

    try:
        classifier = TileClassifier(input_folder=INPUT_FOLDER)
        print(f"Starter klassificering. Resultater gemmes i '{OUTPUT_CSV}'...")
        classifier.process_images(output_csv_file=OUTPUT_CSV, show_images=SHOW_ANNOTATED_IMAGES)
        print(f"\n Færdig. Klassifikationer er gemt i filen: {OUTPUT_CSV}")
    except FileNotFoundError as e:
        print(f"\nFejl: {e}")
        print("Kontroller at input mappen eksisterer og indeholder .jpg filer.")
    except Exception as e:
        print(f"\nfejl: {e}")