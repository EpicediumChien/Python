import os
import glob

# Paths
SOURCE_DIR = r"C:/Git/Python/IMU/data/Time Example"
OUTPUT_DIR = os.path.join(SOURCE_DIR, "fixed_data")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = glob.glob(os.path.join(SOURCE_DIR, "*.csv"))

print(f"Brute-force cleaning {len(files)} files...")

for f in files:
    filename = os.path.basename(f)
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        with open(f, 'r') as file:
            lines = file.readlines()

        fixed_lines = []
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            
            # Check if this is the header row or a data row
            if i == 0:
                # Keep the header as is (it's already 14 columns)
                fixed_lines.append(line)
            else:
                # In the data row, check if parts[1] and parts[2] are the same
                # If they are, we delete parts[2]
                if len(parts) > 2 and parts[1] == parts[2]:
                    # Remove the element at index 2
                    del parts[2]
                    fixed_lines.append(",".join(parts) + "\n")
                else:
                    # If not a duplicate, just keep it
                    fixed_lines.append(line)

        # Write the fixed file
        with open(output_path, 'w') as out_file:
            out_file.writelines(fixed_lines)
            
        print(f"Successfully fixed: {filename}")

    except Exception as e:
        print(f"Failed to fix {filename}: {e}")

print(f"\nCleanup complete. Check the 'fixed_data' folder.")