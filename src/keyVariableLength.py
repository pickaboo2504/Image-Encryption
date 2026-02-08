import math
import numpy as np
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import random
import hashlib

def char_to_binary(c):
    """Convert a character to its 8-bit binary representation."""
    return format(ord(c), '08b')

def binary_to_char(n):
    """Convert an 8-bit binary string to its character representation."""
    return chr(int(n, 2))

def calculate_entropy(data_list):
    """Calculate the entropy of a given data list."""
    length = len(data_list)
    freq = Counter(data_list)
    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
    return entropy

def key_scramble(key, target_length):
    """Expand a key to the target length using scrambling operations."""
    large_key = key
    reverse_temp = key[::-1]
    large_key += reverse_temp

    while len(large_key) < target_length:
        temp = list(key)
        for i in range(len(temp) - 1):
            if temp[i] != temp[i + 1]:
                temp[i], temp[i + 1] = temp[i + 1], temp[i]
                temp_str = ''.join(temp)
                for j in range(len(temp_str)):
                    large_key += temp_str
                    large_key += temp_str[::-1]
                    temp_str = temp_str[1:] + temp_str[0]
                    if len(large_key) >= target_length:
                        break
            if len(large_key) >= target_length:
                break
        key = large_key[:target_length]

    return large_key[:target_length]

def DNADecode(DNA, rule):
    """Decode DNA sequence to binary based on the specified rule."""
    Bits = ''
    if rule == 1:
        for i in range(4):
            if DNA[i] == 'A': Bits += '00'
            elif DNA[i] == 'G': Bits += '01'
            elif DNA[i] == 'C': Bits += '10'
            elif DNA[i] == 'T': Bits += '11'
    elif rule == 2:
        for i in range(4):
            if DNA[i] == 'A': Bits += '00'
            elif DNA[i] == 'C': Bits += '01'
            elif DNA[i] == 'G': Bits += '10'
            elif DNA[i] == 'T': Bits += '11'
    elif rule == 3:
        for i in range(4):
            if DNA[i] == 'T': Bits += '00'
            elif DNA[i] == 'G': Bits += '01'
            elif DNA[i] == 'C': Bits += '10'
            elif DNA[i] == 'A': Bits += '11'
    elif rule == 4:
        for i in range(4):
            if DNA[i] == 'T': Bits += '00'
            elif DNA[i] == 'C': Bits += '01'
            elif DNA[i] == 'G': Bits += '10'
            elif DNA[i] == 'A': Bits += '11'
    elif rule == 5:
        for i in range(4):
            if DNA[i] == 'C': Bits += '00'
            elif DNA[i] == 'T': Bits += '01'
            elif DNA[i] == 'A': Bits += '10'
            elif DNA[i] == 'G': Bits += '11'
    elif rule == 6:
        for i in range(4):
            if DNA[i] == 'C': Bits += '00'
            elif DNA[i] == 'A': Bits += '01'
            elif DNA[i] == 'T': Bits += '10'
            elif DNA[i] == 'G': Bits += '11'
    elif rule == 7:
        for i in range(4):
            if DNA[i] == 'G': Bits += '00'
            elif DNA[i] == 'T': Bits += '01'
            elif DNA[i] == 'A': Bits += '10'
            elif DNA[i] == 'C': Bits += '11'
    elif rule == 8:
        for i in range(4):
            if DNA[i] == 'G': Bits += '00'
            elif DNA[i] == 'A': Bits += '01'
            elif DNA[i] == 'T': Bits += '10'
            elif DNA[i] == 'C': Bits += '11'
    return Bits

def DNAEncode(bits, rule):
    """Encode binary to DNA sequence based on the specified rule."""
    DNA = ''
    if rule == 1:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00': DNA += 'A'
            elif bits[i:i+2] == '01': DNA += 'G'
            elif bits[i:i+2] == '10': DNA += 'C'
            elif bits[i:i+2] == '11': DNA += 'T'
    elif rule == 2:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00': DNA += 'A'
            elif bits[i:i+2] == '01': DNA += 'C'
            elif bits[i:i+2] == '10': DNA += 'G'
            elif bits[i:i+2] == '11': DNA += 'T'
    elif rule == 3:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00': DNA += 'T'
            elif bits[i:i+2] == '01': DNA += 'G'
            elif bits[i:i+2] == '10': DNA += 'C'
            elif bits[i:i+2] == '11': DNA += 'A'
    elif rule == 4:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00': DNA += 'T'
            elif bits[i:i+2] == '01': DNA += 'C'
            elif bits[i:i+2] == '10': DNA += 'G'
            elif bits[i:i+2] == '11': DNA += 'A'
    elif rule == 5:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00': DNA += 'C'
            elif bits[i:i+2] == '01': DNA += 'T'
            elif bits[i:i+2] == '10': DNA += 'A'
            elif bits[i:i+2] == '11': DNA += 'G'
    elif rule == 6:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00': DNA += 'C'
            elif bits[i:i+2] == '01': DNA += 'A'
            elif bits[i:i+2] == '10': DNA += 'T'
            elif bits[i:i+2] == '11': DNA += 'G'
    elif rule == 7:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00': DNA += 'G'
            elif bits[i:i+2] == '01': DNA += 'T'
            elif bits[i:i+2] == '10': DNA += 'A'
            elif bits[i:i+2] == '11': DNA += 'C'
    elif rule == 8:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00': DNA += 'G'
            elif bits[i:i+2] == '01': DNA += 'A'
            elif bits[i:i+2] == '10': DNA += 'T'
            elif bits[i:i+2] == '11': DNA += 'C'
    return DNA

def DNAXOR(DNA1, DNA2):
    """Perform XOR operation on two DNA sequences."""
    DNA = ''
    for i in range(4):
        if ((DNA1[i] == 'A' and DNA2[i] == 'A') or
            (DNA1[i] == 'T' and DNA2[i] == 'T') or
            (DNA1[i] == 'C' and DNA2[i] == 'C') or
            (DNA1[i] == 'G' and DNA2[i] == 'G')):
            DNA += 'A'
        elif ((DNA1[i] == 'A' and DNA2[i] == 'T') or
              (DNA1[i] == 'T' and DNA2[i] == 'A')):
            DNA += 'T'
        elif ((DNA1[i] == 'A' and DNA2[i] == 'C') or
              (DNA1[i] == 'C' and DNA2[i] == 'A')):
            DNA += 'C'
        elif ((DNA1[i] == 'A' and DNA2[i] == 'G') or
              (DNA1[i] == 'G' and DNA2[i] == 'A')):
            DNA += 'G'
        elif ((DNA1[i] == 'T' and DNA2[i] == 'C') or
              (DNA1[i] == 'C' and DNA2[i] == 'T')):
            DNA += 'G'
        elif ((DNA1[i] == 'T' and DNA2[i] == 'G') or
              (DNA1[i] == 'G' and DNA2[i] == 'T')):
            DNA += 'C'
        elif ((DNA1[i] == 'C' and DNA2[i] == 'G') or
              (DNA1[i] == 'G' and DNA2[i] == 'C')):
            DNA += 'T'
    return DNA

def generate_keys(key1_bits):
    seed = int(key1_bits[:8], 2) % 8
    random.seed(seed)
    rule1 = (seed % 8) + 1
    rule2 = ((seed + 1) % 8) + 1
    dna1 = DNAEncode(key1_bits, rule1)
    reverse_key1_bits = key1_bits[::-1]
    dna2 = DNAEncode(reverse_key1_bits, rule2)
    xored = DNAXOR(dna1, dna2)
    key2 = DNADecode(xored, rule=1)
    
    rule3 = ((seed + 2) % 8) + 1
    rule4 = ((seed + 3) % 8) + 1
    dna3 = DNAEncode(key1_bits[-8:], rule3)
    dna4 = DNAEncode(key2[-8:], rule4)
    xored2 = DNAXOR(dna3, dna4)
    key3 = DNADecode(xored2, rule=1)
    
    return key2, key3

def shift_pixels(image_array, shift_value):
    shifted_image = np.roll(image_array, shift_value, axis=(0, 1))
    return shifted_image

def generate_shift_value(key1, key2, key3):
    """Generate an unpredictable shift value between 1-100 based on keys."""
    combined_key = key1 + key2 + key3
    hash_value = hashlib.sha256(combined_key.encode()).hexdigest()
    shift_value = int(hash_value[:8], 16) % 100 + 1
    return shift_value

def encryption(key1, key2, key3, image_path):
    """Encrypt an image using triple-layer DNA encoding."""
    image = Image.open(image_path)
    image_array = np.array(image)

    shift_value = generate_shift_value(key1, key2, key3)
    image_array = shift_pixels(image_array, shift_value)
    
    height, width, channels = image_array.shape
    processed_image = np.zeros(shape=(height, width, channels), dtype=np.uint8)

    index = 0
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if index < len(key1):
                    rule1, rule2, rule3 = (index % 8) + 1, ((index + 1) % 8) + 1, ((index + 2) % 8) + 1
                    key_segment1 = key1[index*8:(index+1)*8]
                    key_segment2 = key2[index*8:(index+1)*8]
                    key_segment3 = key3[index*8:(index+1)*8]
                    
                    original_bin = format(image_array[i, j, k], '08b')
                    dna1 = DNAEncode(original_bin, rule1)
                    xored1 = DNAXOR(dna1, DNAEncode(key_segment1, rule1))
                    temp1 = DNADecode(xored1, rule1)
                    
                    dna2 = DNAEncode(temp1, rule2)
                    xored2 = DNAXOR(dna2, DNAEncode(key_segment2, rule2))
                    temp2 = DNADecode(xored2, rule2)
                    
                    dna3 = DNAEncode(temp2, rule3)
                    xored3 = DNAXOR(dna3, DNAEncode(key_segment3, rule3))
                    processed_image[i, j, k] = int(DNADecode(xored3, rule3), 2)
                    
                    index += 1
    return processed_image

def plot_results(original_image, cipher_image, key_name):
    """Plot and save results for a given key."""
    # Convert images to arrays
    original_array = np.array(original_image)
    cipher_array = cipher_image
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title(f"Original Image ({key_name})")
    plt.axis('off')
    
    # Encrypted image
    plt.subplot(2, 2, 2)
    plt.imshow(cipher_image)
    plt.title(f"Encrypted Image ({key_name})")
    plt.axis('off')
    
    # Original histogram
    plt.subplot(2, 2, 3)
    plt.hist(original_array.flatten(), bins=256, color='blue', alpha=0.7)
    plt.title(f"Original Histogram ({key_name})")
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    # Encrypted histogram
    plt.subplot(2, 2, 4)
    plt.hist(cipher_array.flatten(), bins=256, color='red', alpha=0.7)
    plt.title(f"Encrypted Histogram ({key_name})")
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print entropy
    original_entropy = calculate_entropy(original_array.flatten())
    cipher_entropy = calculate_entropy(cipher_array.flatten())
    print(f"\n{key_name} Results:")
    print(f"Original Image Entropy: {original_entropy:.4f}")
    print(f"Encrypted Image Entropy: {cipher_entropy:.4f}")

def plot_and_save_results(original_image, cipher_image1, cipher_image2, key1_name, key2_name, save_path):
    """Plot and save results for both keys in one figure."""
    # Convert images to arrays
    original_array = np.array(original_image)
    cipher_array1 = cipher_image1
    cipher_array2 = cipher_image2
    
    # Create figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # First encrypted image
    axs[0, 0].imshow(cipher_image1)
    axs[0, 0].set_title(f"Encrypted Image ({key1_name})")
    axs[0, 0].axis('off')
    
    # First encrypted histogram
    axs[0, 1].hist(cipher_array1.flatten(), bins=256, color='red', alpha=0.7)
    axs[0, 1].set_title(f"Histogram ({key1_name})")
    axs[0, 1].set_xlabel('Pixel Value')
    axs[0, 1].set_ylabel('Frequency')
    
    # Second encrypted image
    axs[1, 0].imshow(cipher_image2)
    axs[1, 0].set_title(f"Encrypted Image ({key2_name})")
    axs[1, 0].axis('off')
    
    # Second encrypted histogram
    axs[1, 1].hist(cipher_array2.flatten(), bins=256, color='green', alpha=0.7)
    axs[1, 1].set_title(f"Histogram ({key2_name})")
    axs[1, 1].set_xlabel('Pixel Value')
    axs[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # Calculate and print entropy
    original_entropy = calculate_entropy(original_array.flatten())
    cipher_entropy1 = calculate_entropy(cipher_array1.flatten())
    cipher_entropy2 = calculate_entropy(cipher_array2.flatten())
    
    print("\nEntropy Results:")
    print(f"Original Image Entropy: {original_entropy:.4f}")
    print(f"Encrypted Image ({key1_name}) Entropy: {cipher_entropy1:.4f}")
    print(f"Encrypted Image ({key2_name}) Entropy: {cipher_entropy2:.4f}")
    
    plt.show()

def process_key(key, image_path):
    """Process a single key and return encrypted image."""
    image = Image.open(image_path)
    image_array = np.array(image)
    
    image_row, image_col, channel = image_array.shape
    target_length = image_row * image_col * channel * 8

    key1_bits = ''.join(char_to_binary(c) for c in key)
    key2_bits, key3_bits = generate_keys(key1_bits)

    key1_bits = key_scramble(key1_bits, target_length)
    key2_bits = key_scramble(key2_bits, target_length)
    key3_bits = key_scramble(key3_bits, target_length)
    
    cipher_image = encryption(key1_bits, key2_bits, key3_bits, image_path)
    
    return cipher_image

def main():
    image_path = 'khalid/airplane256.png'  # Replace with your image path
    save_path = 'cipher_comparison.png'    # Output file name
    
    # Load original image
    original_image = Image.open(image_path)
    
    # Process first key
    key1 = "SecureDataMatters"
    cipher_image1 = process_key(key1, image_path)
    
    # Process second key
    key2 = "SecureDataMattersInModernEncryption"
    cipher_image2 = process_key(key2, image_path)
    
    # Plot and save results
    plot_and_save_results(original_image, cipher_image1, cipher_image2, key1, key2, save_path)

if __name__ == "__main__":
    main()