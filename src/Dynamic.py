import math
import numpy as np
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import random

def char_to_binary(c):
    return format(ord(c), '08b')

def binary_to_char(n):
    return int(n, 2)

def calculate_entropy(data_list):
    length = len(data_list)
    freq = Counter(data_list)
    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
    return entropy

def key_scramble(key, target_length):
    large_key = key
    reverse_temp = key[::-1]
    large_key += reverse_temp

    while len(large_key) < target_length:
        temp = list(key)
        for i in range(len(temp) - 1):
            if temp[i] != temp[i + 1]:
                temp[i], temp[i + 1] = temp[i + 1], temp[i]
                temp_str = ''.join(temp)
                large_key += temp_str + temp_str[::-1]
                if len(large_key) >= target_length:
                    break
        key = large_key[:target_length]

    return large_key[:target_length]

def DNADecode(DNA, rule):
    Bits = ''
    j = 0
    if rule == 1:
        for i in range(4):
            if DNA[i] == 'A':
                Bits += '00'
            elif DNA[i] == 'G':
                Bits += '01'
            elif DNA[i] == 'C':
                Bits += '10'
            elif DNA[i] == 'T':
                Bits += '11'
            j += 2
    elif rule == 2:
        for i in range(4):
            if DNA[i] == 'A':
                Bits += '00'
            elif DNA[i] == 'C':
                Bits += '01'
            elif DNA[i] == 'G':
                Bits += '10'
            elif DNA[i] == 'T':
                Bits += '11'
            j += 2
    elif rule == 3:
        for i in range(4):
            if DNA[i] == 'T':
                Bits += '00'
            elif DNA[i] == 'G':
                Bits += '01'
            elif DNA[i] == 'C':
                Bits += '10'
            elif DNA[i] == 'A':
                Bits += '11'
            j += 2
    elif rule == 4:
        for i in range(4):
            if DNA[i] == 'T':
                Bits += '00'
            elif DNA[i] == 'C':
                Bits += '01'
            elif DNA[i] == 'G':
                Bits += '10'
            elif DNA[i] == 'A':
                Bits += '11'
            j += 2
    elif rule == 5:
        for i in range(4):
            if DNA[i] == 'C':
                Bits += '00'
            elif DNA[i] == 'T':
                Bits += '01'
            elif DNA[i] == 'A':
                Bits += '10'
            elif DNA[i] == 'G':
                Bits += '11'
            j += 2
    elif rule == 6:
        for i in range(4):
            if DNA[i] == 'C':
                Bits += '00'
            elif DNA[i] == 'A':
                Bits += '01'
            elif DNA[i] == 'T':
                Bits += '10'
            elif DNA[i] == 'G':
                Bits += '11'
            j += 2
    elif rule == 7:
        for i in range(4):
            if DNA[i] == 'G':
                Bits += '00'
            elif DNA[i] == 'T':
                Bits += '01'
            elif DNA[i] == 'A':
                Bits += '10'
            elif DNA[i] == 'C':
                Bits += '11'
            j += 2
    elif rule == 8:
        for i in range(4):
            if DNA[i] == 'G':
                Bits += '00'
            elif DNA[i] == 'A':
                Bits += '01'
            elif DNA[i] == 'T':
                Bits += '10'
            elif DNA[i] == 'C':
                Bits += '11'
            j += 2
    return Bits

def DNAEncode(bits, rule):
    DNA = ''
    j = 0
    if rule == 1:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00':
                DNA += 'A'
            elif bits[i:i+2] == '01':
                DNA += 'G'
            elif bits[i:i+2] == '10':
                DNA += 'C'
            elif bits[i:i+2] == '11':
                DNA += 'T'
            j += 1
    elif rule == 2:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00':
                DNA += 'A'
            elif bits[i:i+2] == '01':
                DNA += 'C'
            elif bits[i:i+2] == '10':
                DNA += 'G'
            elif bits[i:i+2] == '11':
                DNA += 'T'
            j += 1
    elif rule == 3:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00':
                DNA += 'T'
            elif bits[i:i+2] == '01':
                DNA += 'G'
            elif bits[i:i+2] == '10':
                DNA += 'C'
            elif bits[i:i+2] == '11':
                DNA += 'A'
            j += 1
    elif rule == 4:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00':
                DNA += 'T'
            elif bits[i:i+2] == '01':
                DNA += 'C'
            elif bits[i:i+2] == '10':
                DNA += 'G'
            elif bits[i:i+2] == '11':
                DNA += 'A'
            j += 1
    elif rule == 5:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00':
                DNA += 'C'
            elif bits[i:i+2] == '01':
                DNA += 'T'
            elif bits[i:i+2] == '10':
                DNA += 'A'
            elif bits[i:i+2] == '11':
                DNA += 'G'
            j += 1
    elif rule == 6:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00':
                DNA += 'C'
            elif bits[i:i+2] == '01':
                DNA += 'A'
            elif bits[i:i+2] == '10':
                DNA += 'T'
            elif bits[i:i+2] == '11':
                DNA += 'G'
            j += 1
    elif rule == 7:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00':
                DNA += 'G'
            elif bits[i:i+2] == '01':
                DNA += 'T'
            elif bits[i:i+2] == '10':
                DNA += 'A'
            elif bits[i:i+2] == '11':
                DNA += 'C'
            j += 1
    elif rule == 8:
        for i in range(0, 8, 2):
            if bits[i:i+2] == '00':
                DNA += 'G'
            elif bits[i:i+2] == '01':
                DNA += 'A'
            elif bits[i:i+2] == '10':
                DNA += 'T'
            elif bits[i:i+2] == '11':
                DNA += 'C'
            j += 1
    return DNA

def DNAXOR(DNA1, DNA2):
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

def new_key(key1_bits):
    dna1 = DNAEncode(key1_bits,rule=1)
    reverse_key1_bits = key1_bits[::-1]
    dna2 = DNAEncode(reverse_key1_bits,rule=1)
    xored = DNAXOR(dna1, dna2)
    key2 = DNADecode(xored,rule=1)

    return key2

# Function to generate random DNA encoding rules for each pixel
def generate_random_rules(height, width, channels):
    return np.random.randint(1, 9, size=(height, width, channels))

# Modify encryption function to use dynamic DNA encoding rules
def encryption(key1, key2, image_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    height, width, channels = image_array.shape
    processed_image = np.zeros((height, width, channels), dtype=np.uint8)

    # Generate a random DNA encoding rule for each pixel
    rules = generate_random_rules(height, width, channels)

    index = 0
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if index < len(key1):
                    rule = rules[i, j, k]  # Get the randomly assigned rule for this pixel
                    key_segment1 = key1[index*8:(index+1)*8]
                    key_segment2 = key2[index*8:(index+1)*8]

                    original_bin = format(image_array[i, j, k], '08b')
                    dna1 = DNAEncode(original_bin, rule)
                    xored1 = DNAXOR(dna1, DNAEncode(key_segment1, rule))
                    dna2 = DNAEncode(DNADecode(xored1, rule), rule)
                    xored2 = DNAXOR(dna2, DNAEncode(key_segment2, rule))

                    processed_image[i, j, k] = int(DNADecode(xored2, rule), 2)
                    index += 1

    return processed_image, rules  # Return encrypted image and stored rules

# Modify decryption function to use stored rules
def decryption(key1, key2, encrypted_image, rules):
    cipher_array = np.array(encrypted_image)
    height, width, channels = cipher_array.shape
    decrypted_image = np.zeros((height, width, channels), dtype=np.uint8)

    index = 0
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if index < len(key1):
                    rule = rules[i, j, k]  # Use the same rule assigned during encryption
                    key_segment1 = key1[index*8:(index+1)*8]
                    key_segment2 = key2[index*8:(index+1)*8]

                    encrypted_bin = format(cipher_array[i, j, k], '08b')
                    dna2 = DNAEncode(encrypted_bin, rule)
                    xored2 = DNAXOR(dna2, DNAEncode(key_segment2, rule))
                    dna1 = DNAEncode(DNADecode(xored2, rule), rule)
                    xored1 = DNAXOR(dna1, DNAEncode(key_segment1, rule))

                    decrypted_image[i, j, k] = int(DNADecode(xored1, rule), 2)
                    index += 1

    return decrypted_image

# Modify main function to store and use random rules
def main():
    image_path = 'image/Lena512.png'
    image = Image.open(image_path)
    image_array = np.array(image)

    image_row, image_col, channel = image_array.shape
    target_length = image_row * image_col * channel * 8

    key = "Bangladesh"
    key1_bits = ''.join(char_to_binary(c) for c in key)
    key2_bits = new_key(key1_bits)

    key1_bits = key_scramble(key1_bits, target_length)
    key2_bits = key_scramble(key2_bits, target_length)

    cipherImage, rules = encryption(key1_bits, key2_bits, image_path)

    plt.imshow(cipherImage)
    plt.title("Encrypted Image")
    plt.show()

    original_1d = image_array.flatten()
    print("Entropy original image:", calculate_entropy(original_1d))
    plt.hist(original_1d, bins=256)
    plt.show()

    cipherImage_1d = cipherImage.flatten()
    print("Entropy of New Algorithm:", calculate_entropy(cipherImage_1d))
    plt.hist(cipherImage_1d, bins=256)
    plt.show()

    decrypted_image = decryption(key1_bits, key2_bits, cipherImage, rules)
    plt.imshow(decrypted_image)
    plt.title("Decrypted Image")
    plt.show()

main()