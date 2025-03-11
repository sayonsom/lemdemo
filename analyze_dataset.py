import json

# Load synthetic dataset
with open('event_dataset.json', 'r') as file:
    dataset = json.load(file)

print(f"Number of sequences: {len(dataset)}")
print(f"Sequence lengths: {[len(seq) for seq in dataset[:5]]}")

# Check if all sequences have the same length
sequence_lengths = [len(seq) for seq in dataset]
if len(set(sequence_lengths)) > 1:
    print(f"WARNING: Not all sequences have the same length!")
    print(f"Unique sequence lengths: {set(sequence_lengths)}")
    print(f"Min length: {min(sequence_lengths)}, Max length: {max(sequence_lengths)}")
    
    # Find the first sequence with a different length
    expected_length = sequence_lengths[0]
    for i, length in enumerate(sequence_lengths):
        if length != expected_length:
            print(f"First different sequence at index {i}: length {length} vs expected {expected_length}")
            break
else:
    print(f"All sequences have the same length: {sequence_lengths[0]}") 