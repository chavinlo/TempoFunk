import os
import csv
from collections import Counter

# Create a dictionary to store the word frequencies
word_freqs = {}

# Get a list of all the txt files in the folder
folder_path = "/workspace/disk/webvid/raw"
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Iterate over each file and count the word frequencies
for filename in file_list:
    with open(os.path.join(folder_path, filename), 'r') as file:
        # Read in the file contents and split into words
        words = file.read().lower().split()
        # Count the frequency of each word in the file
        freqs = Counter(words)
        # Update the word frequency dictionary with the counts from this file
        for word, count in freqs.items():
            if word not in word_freqs:
                word_freqs[word] = [count]
            else:
                word_freqs[word].append(count)

# Sort the word frequency dictionary by the total count of each word
sorted_word_freqs = sorted(word_freqs.items(), key=lambda x: sum(x[1]), reverse=True)

# Write the results to a CSV file
with open('word_freqs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(['Word', 'Total Count', 'File Counts'])
    # Write each row of data
    for word, counts in sorted_word_freqs:
        writer.writerow([word, sum(counts), counts])
