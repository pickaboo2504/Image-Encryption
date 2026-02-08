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
    # Convert part of hash to integer in range 1-100
    shift_value = int(hash_value[:8], 16) % 100 + 1  # Ensures value is between 1-100
    return shift_value

def encryption(key1, key2, key3, image_array):
    """Encrypt an image using triple-layer DNA encoding."""
    height, width, channels = image_array.shape
    shift_value = generate_shift_value(key1, key2, key3)
    image_array = shift_pixels(image_array, shift_value)
    
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

def decryption(key1, key2, key3, cipher_array):
    """Decrypt an image using triple-layer DNA decoding."""
    height, width, channels = cipher_array.shape
    decrypted_image = np.zeros(shape=(height, width, channels), dtype=np.uint8)

    index = 0
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if index < len(key1):
                    rule1, rule2, rule3 = (index % 8) + 1, ((index + 1) % 8) + 1, ((index + 2) % 8) + 1
                    key_segment1 = key1[index*8:(index+1)*8]
                    key_segment2 = key2[index*8:(index+1)*8]
                    key_segment3 = key3[index*8:(index+1)*8]
                    
                    encrypted_bin = format(cipher_array[i, j, k], '08b')
                    dna3 = DNAEncode(encrypted_bin, rule3)
                    xored3 = DNAXOR(dna3, DNAEncode(key_segment3, rule3))
                    temp3 = DNADecode(xored3, rule3)
                    
                    dna2 = DNAEncode(temp3, rule2)
                    xored2 = DNAXOR(dna2, DNAEncode(key_segment2, rule2))
                    temp2 = DNADecode(xored2, rule2)
                    
                    dna1 = DNAEncode(temp2, rule1)
                    xored1 = DNAXOR(dna1, DNAEncode(key_segment1, rule1))
                    decrypted_image[i, j, k] = int(DNADecode(xored1, rule1), 2)
                    
                    index += 1
    
    shift_value = generate_shift_value(key1, key2, key3)
    decrypted_image = shift_pixels(decrypted_image, -shift_value)
    return decrypted_image

def modify_image_portion(image, start_x, start_y, size_x, size_y, change_value=5):
    """
    Modify a small portion of pixels in the image
    
    Args:
        image: The original image as numpy array
        start_x, start_y: Starting coordinates of the portion to modify
        size_x, size_y: Size of the area to modify
        change_value: Value to add to the pixels
    
    Returns:
        Modified image as numpy array
    """
    modified_image = np.copy(image)
    
    # Define the region to modify
    end_x = min(start_x + size_x, image.shape[0])
    end_y = min(start_y + size_y, image.shape[1])
    
    # Add the change value to the pixels in the region
    for i in range(start_x, end_x):
        for j in range(start_y, end_y):
            for c in range(image.shape[2]):
                # Add the change value and ensure it's within 0-255
                modified_image[i, j, c] = np.clip(image[i, j, c] + change_value, 0, 255)
    
    return modified_image

def calculate_NPCR(cipher1, cipher2):
    """
    Calculate Number of Pixel Change Rate (NPCR)
    
    Args:
        cipher1: First cipher image
        cipher2: Second cipher image (after small change in plain image)
    
    Returns:
        NPCR value as percentage
    """
    height, width, channels = cipher1.shape
    diff_count = 0
    total_pixels = height * width * channels
    
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if cipher1[i, j, k] != cipher2[i, j, k]:
                    diff_count += 1
    
    npcr = (diff_count / total_pixels) * 100
    return npcr

def calculate_UACI(cipher1, cipher2):
    """
    Calculate Unified Average Changing Intensity (UACI)
    
    Args:
        cipher1: First cipher image
        cipher2: Second cipher image (after small change in plain image)
    
    Returns:
        UACI value as percentage
    """
    height, width, channels = cipher1.shape
    sum_diff = 0
    total_pixels = height * width * channels
    
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                sum_diff += abs(int(cipher1[i, j, k]) - int(cipher2[i, j, k]))
    
    uaci = (sum_diff / (255 * total_pixels)) * 100
    return uaci

def differential_attack_analysis(image_path, key, portion_size=10):
    """
    Perform differential attack analysis by changing a small portion of pixels
    and calculating NPCR and UACI metrics
    
    Args:
        image_path: Path to the original image
        key: Encryption key
        portion_size: Size of the square portion to modify
    
    Returns:
        NPCR and UACI values
    """
    # Load original image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Prepare keys
    key1_bits = ''.join(char_to_binary(c) for c in key)
    key2_bits, key3_bits = generate_keys(key1_bits)
    
    # Calculate target length for key expansion
    height, width, channels = image_array.shape
    target_length = height * width * channels * 8
    
    # Expand keys
    key1_bits = key_scramble(key1_bits, target_length)
    key2_bits = key_scramble(key2_bits, target_length)
    key3_bits = key_scramble(key3_bits, target_length)
    
    # Encrypt original image
    cipher1 = encryption(key1_bits, key2_bits, key3_bits, image_array)
    
    # Modify a portion of the original image
    start_x = height // 4  # Start from 1/4 of the height
    start_y = width // 4   # Start from 1/4 of the width
    
    # Modify image
    modified_image = modify_image_portion(image_array, start_x, start_y, portion_size, portion_size, change_value=5)
    
    # Encrypt modified image
    cipher2 = encryption(key1_bits, key2_bits, key3_bits, modified_image)
    
    # Calculate NPCR and UACI
    npcr = calculate_NPCR(cipher1, cipher2)
    uaci = calculate_UACI(cipher1, cipher2)
    
    return cipher1, cipher2, modified_image, npcr, uaci

def main():
    """Main function to run the differential attack analysis."""
    image_path = 'khalid/Lena256.png'  # Update with your image path
    key = "Bangladesh"
    portion_size = 10  # Size of the portion to modify
    
    # Perform differential attack analysis
    cipher1, cipher2, modified_image, npcr, uaci = differential_attack_analysis(image_path, key, portion_size)
    
    # Print results
    print(f"NPCR (Number of Pixel Change Rate): {npcr:.4f}%")
    print(f"UACI (Unified Average Changing Intensity): {uaci:.4f}%")
    print(f"Ideal NPCR should be close to 99.6%")
    print(f"Ideal UACI should be close to 33.4%")
    
    # Display images
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    image = Image.open(image_path)
    image_array = np.array(image)
    axes[0].imshow(image_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Modified image
    axes[1].imshow(modified_image)
    axes[1].set_title(f"Modified Image\n(changed {portion_size}×{portion_size} pixels)")
    axes[1].axis('off')
    
    # First cipher
    axes[2].imshow(cipher1)
    axes[2].set_title("Cipher of Original Image")
    axes[2].axis('off')
    
    # Second cipher
    axes[3].imshow(cipher2)
    axes[3].set_title("Cipher of Modified Image")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram of first cipher
    axes[0].hist(cipher1.flatten(), bins=256, color='blue', alpha=0.7)
    axes[0].set_title("Histogram of Original Cipher")
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Frequency")
    
    # Histogram of second cipher
    axes[1].hist(cipher2.flatten(), bins=256, color='red', alpha=0.7)
    axes[1].set_title("Histogram of Modified Cipher")
    axes[1].set_xlabel("Pixel Value")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()