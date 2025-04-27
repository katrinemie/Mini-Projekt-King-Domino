import csv

def convert_crowns_to_ground_truth(input_csv, output_csv):
    data = []
    
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row['label']
            if ' ' in label:
                terrain, crowns = label.rsplit(' ', 1)  
            else:
                terrain = label
                crowns = 0  #Hvis ingen kroner nævnt

            data.append({
                "image_id": row["image_id"],
                "x": row["x"],
                "y": row["y"],
                "terrain": terrain,
                "crowns": crowns
            })

    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["image_id", "x", "y", "terrain", "crowns"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"færdig. Ny fil gemt: {output_csv}")

if __name__ == "__main__":
   
    input_csv = "crowns_annotations.csv"
    output_csv = "ground_truth2_brug_ikke.csv"
    
    convert_crowns_to_ground_truth(input_csv, output_csv)
