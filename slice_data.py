print("🔪 Slicing 15,000 lines from the massive CSV...")

try:
    # Open the giant file in 'read' mode, and a new text file in 'write' mode
    with open('enron.csv', 'r', encoding='utf-8', errors='ignore') as infile:
        with open('massive_enron.txt', 'w', encoding='utf-8') as outfile:
            
            # Read exactly 15,000 lines silently and save them
            for i in range(15000):
                line = infile.readline()
                if not line: 
                    break # Stop if we reach the end early
                outfile.write(line)

    print("✅ Success! 'massive_enron.txt' has been created safely.")
except FileNotFoundError:
    print("❌ Error: Could not find 'enron.csv'. Make sure it is in this folder!")
