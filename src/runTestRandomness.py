import math
import numpy as np
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import random
import hashlib
import warnings
from scipy.stats import norm

# Suppress overflow warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

def decryption(key1, key2, key3, encrypted_image):
    """Decrypt an image using triple-layer DNA decoding."""
    cipher_array = np.array(encrypted_image)

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


def runs_test(binary_sequence):
    """
    Perform the runs test for randomness on a binary sequence.
    
    Args:
        binary_sequence: A string or list of binary digits ('0's and '1's)
        
    Returns:
        A tuple containing:
        - result: 0 if random, 1 if not random
        - p_value: The p-value of the test
        - randomness: "Satisfied" if random, "Not satisfied" otherwise
    """
    n = len(binary_sequence)
    if n == 0:
        return 1, 0.0, "Not satisfied"
    
    # Count the number of ones and zeros
    n1 = binary_sequence.count('1')
    n0 = binary_sequence.count('0')
    
    # Calculate the expected number of runs
    expected_runs = (2 * n0 * n1) / (n0 + n1) + 1
    
    # Count the actual number of runs
    runs = 1
    for i in range(1, len(binary_sequence)):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1
    
    # Calculate the standard deviation
    numerator = 2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)
    denominator = (n0 + n1)**2 * (n0 + n1 - 1)
    std_dev = np.sqrt(numerator / denominator)
    
    # Calculate the z-score
    if std_dev == 0:
        z = 0
    else:
        z = (runs - expected_runs) / std_dev
    
    # Calculate the p-value (two-tailed test)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    # Determine if the sequence is random (using common alpha=0.05)
    if p_value >= 0.05:
        return 0, p_value, "Satisfied"
    else:
        return 1, p_value, "Not satisfied"

def test_encrypted_image_randomness(cipher_image):
    """
    Test the randomness of an encrypted image by analyzing each color channel.
    
    Args:
        cipher_image: The encrypted image as a numpy array
        
    Returns:
        A dictionary with results for each color channel
    """
    results = {}
    
    # Convert each channel to binary and perform runs test
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
        # Extract channel data and convert to binary string
        channel_data = cipher_image[:, :, channel_idx].flatten()
        binary_str = ''.join([format(pixel, '08b') for pixel in channel_data])
        
        # Perform runs test
        result, p_value, randomness = runs_test(binary_str)
        
        results[channel_name] = {
            'Result': result,
            'P-value': p_value,
            'Randomness': randomness
        }
    
    return results

# Modified main function to include randomness testing
def main():
    image_path = 'khalid/Baboon512.png' 
    image = Image.open(image_path)
    image_array = np.array(image)
    
    image_row, image_col, channel = image_array.shape
    target_length = image_row * image_col * channel * 8

    key = "Bangladesh"
    key1_bits = ''.join(char_to_binary(c) for c in key)

    key2_bits, key3_bits = generate_keys(key1_bits)

    key1_bits = key_scramble(key1_bits, target_length)
    key2_bits = key_scramble(key2_bits, target_length)
    key3_bits = key_scramble(key3_bits, target_length)
    
    cipherImage = encryption(key1_bits, key2_bits, key3_bits, image_path)
    
    plt.imshow(cipherImage)
    plt.title("Encrypted Image")
    plt.show()
    
    # Test randomness of the encrypted image
    randomness_results = test_encrypted_image_randomness(cipherImage)
    
    # Print results in table format similar to your example
    print("\nResults for randomness test for encrypted images")
    print("{:<15} {:<6} {:<8} {:<10}".format("Images (256 × 256)", "Plane", "Result", "Randomness"))
    print("-" * 45)
    
    image_name = "Lena"
    for channel in ['R', 'G', 'B']:
        result = randomness_results[channel]['Result']
        randomness = randomness_results[channel]['Randomness']
        if channel == 'R':
            print("{:<15} {:<6} {:<8} {:<10}".format(image_name, channel, result, randomness))
        else:
            print("{:<15} {:<6} {:<8} {:<10}".format("", channel, result, randomness))
    
    original_1d = image_array.flatten()
    print("\nOriginal Image Entropy:", calculate_entropy(original_1d)) 
    plt.hist(original_1d, bins=256)
    plt.title("Original Image Histogram")
    plt.show()

    cipherImage_1d = cipherImage.flatten()
    print("Encrypted Image Entropy:", calculate_entropy(cipherImage_1d))
    plt.hist(cipherImage_1d, bins=256)
    plt.title("Encrypted Image Histogram")
    plt.show() 

    decrypted_image = decryption(key1_bits, key2_bits, key3_bits, cipherImage)
    
    plt.imshow(decrypted_image)
    plt.title("Decrypted Image")
    plt.show()

if __name__ == "__main__":
    main()