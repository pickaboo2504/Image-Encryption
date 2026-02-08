# import math
# import numpy as np
# from collections import Counter
# from PIL import Image
# import matplotlib.pyplot as plt
# import random
# import hashlib

# def char_to_binary(c):
#     """Convert a character to its 8-bit binary representation."""
#     return format(ord(c), '08b')

# def binary_to_char(n):
#     """Convert an 8-bit binary string to its character representation."""
#     return chr(int(n, 2))

# def calculate_entropy(data_list):
#     """Calculate the entropy of a given data list."""
#     length = len(data_list)
#     freq = Counter(data_list)
#     entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
#     return entropy

# def key_scramble(key, target_length):
#     """Expand a key to the target length using scrambling operations."""
#     large_key = key
#     reverse_temp = key[::-1]
#     large_key += reverse_temp

#     while len(large_key) < target_length:
#         temp = list(key)
#         for i in range(len(temp) - 1):
#             if temp[i] != temp[i + 1]:
#                 temp[i], temp[i + 1] = temp[i + 1], temp[i]
#                 temp_str = ''.join(temp)
#                 for j in range(len(temp_str)):
#                     large_key += temp_str
#                     large_key += temp_str[::-1]
#                     temp_str = temp_str[1:] + temp_str[0]
#                     if len(large_key) >= target_length:
#                         break
#             if len(large_key) >= target_length:
#                 break
#         key = large_key[:target_length]

#     return large_key[:target_length]

# def DNADecode(DNA, rule):
#     """Decode DNA sequence to binary based on the specified rule."""
#     Bits = ''
#     if rule == 1:
#         for i in range(4):
#             if DNA[i] == 'A': Bits += '00'
#             elif DNA[i] == 'G': Bits += '01'
#             elif DNA[i] == 'C': Bits += '10'
#             elif DNA[i] == 'T': Bits += '11'
#     elif rule == 2:
#         for i in range(4):
#             if DNA[i] == 'A': Bits += '00'
#             elif DNA[i] == 'C': Bits += '01'
#             elif DNA[i] == 'G': Bits += '10'
#             elif DNA[i] == 'T': Bits += '11'
#     elif rule == 3:
#         for i in range(4):
#             if DNA[i] == 'T': Bits += '00'
#             elif DNA[i] == 'G': Bits += '01'
#             elif DNA[i] == 'C': Bits += '10'
#             elif DNA[i] == 'A': Bits += '11'
#     elif rule == 4:
#         for i in range(4):
#             if DNA[i] == 'T': Bits += '00'
#             elif DNA[i] == 'C': Bits += '01'
#             elif DNA[i] == 'G': Bits += '10'
#             elif DNA[i] == 'A': Bits += '11'
#     elif rule == 5:
#         for i in range(4):
#             if DNA[i] == 'C': Bits += '00'
#             elif DNA[i] == 'T': Bits += '01'
#             elif DNA[i] == 'A': Bits += '10'
#             elif DNA[i] == 'G': Bits += '11'
#     elif rule == 6:
#         for i in range(4):
#             if DNA[i] == 'C': Bits += '00'
#             elif DNA[i] == 'A': Bits += '01'
#             elif DNA[i] == 'T': Bits += '10'
#             elif DNA[i] == 'G': Bits += '11'
#     elif rule == 7:
#         for i in range(4):
#             if DNA[i] == 'G': Bits += '00'
#             elif DNA[i] == 'T': Bits += '01'
#             elif DNA[i] == 'A': Bits += '10'
#             elif DNA[i] == 'C': Bits += '11'
#     elif rule == 8:
#         for i in range(4):
#             if DNA[i] == 'G': Bits += '00'
#             elif DNA[i] == 'A': Bits += '01'
#             elif DNA[i] == 'T': Bits += '10'
#             elif DNA[i] == 'C': Bits += '11'
#     return Bits

# def DNAEncode(bits, rule):
#     """Encode binary to DNA sequence based on the specified rule."""
#     DNA = ''
#     if rule == 1:
#         for i in range(0, 8, 2):
#             if bits[i:i+2] == '00': DNA += 'A'
#             elif bits[i:i+2] == '01': DNA += 'G'
#             elif bits[i:i+2] == '10': DNA += 'C'
#             elif bits[i:i+2] == '11': DNA += 'T'
#     elif rule == 2:
#         for i in range(0, 8, 2):
#             if bits[i:i+2] == '00': DNA += 'A'
#             elif bits[i:i+2] == '01': DNA += 'C'
#             elif bits[i:i+2] == '10': DNA += 'G'
#             elif bits[i:i+2] == '11': DNA += 'T'
#     elif rule == 3:
#         for i in range(0, 8, 2):
#             if bits[i:i+2] == '00': DNA += 'T'
#             elif bits[i:i+2] == '01': DNA += 'G'
#             elif bits[i:i+2] == '10': DNA += 'C'
#             elif bits[i:i+2] == '11': DNA += 'A'
#     elif rule == 4:
#         for i in range(0, 8, 2):
#             if bits[i:i+2] == '00': DNA += 'T'
#             elif bits[i:i+2] == '01': DNA += 'C'
#             elif bits[i:i+2] == '10': DNA += 'G'
#             elif bits[i:i+2] == '11': DNA += 'A'
#     elif rule == 5:
#         for i in range(0, 8, 2):
#             if bits[i:i+2] == '00': DNA += 'C'
#             elif bits[i:i+2] == '01': DNA += 'T'
#             elif bits[i:i+2] == '10': DNA += 'A'
#             elif bits[i:i+2] == '11': DNA += 'G'
#     elif rule == 6:
#         for i in range(0, 8, 2):
#             if bits[i:i+2] == '00': DNA += 'C'
#             elif bits[i:i+2] == '01': DNA += 'A'
#             elif bits[i:i+2] == '10': DNA += 'T'
#             elif bits[i:i+2] == '11': DNA += 'G'
#     elif rule == 7:
#         for i in range(0, 8, 2):
#             if bits[i:i+2] == '00': DNA += 'G'
#             elif bits[i:i+2] == '01': DNA += 'T'
#             elif bits[i:i+2] == '10': DNA += 'A'
#             elif bits[i:i+2] == '11': DNA += 'C'
#     elif rule == 8:
#         for i in range(0, 8, 2):
#             if bits[i:i+2] == '00': DNA += 'G'
#             elif bits[i:i+2] == '01': DNA += 'A'
#             elif bits[i:i+2] == '10': DNA += 'T'
#             elif bits[i:i+2] == '11': DNA += 'C'
#     return DNA

# def DNAXOR(DNA1, DNA2):
#     """Perform XOR operation on two DNA sequences."""
#     DNA = ''
#     for i in range(4):
#         if ((DNA1[i] == 'A' and DNA2[i] == 'A') or
#             (DNA1[i] == 'T' and DNA2[i] == 'T') or
#             (DNA1[i] == 'C' and DNA2[i] == 'C') or
#             (DNA1[i] == 'G' and DNA2[i] == 'G')):
#             DNA += 'A'
#         elif ((DNA1[i] == 'A' and DNA2[i] == 'T') or
#               (DNA1[i] == 'T' and DNA2[i] == 'A')):
#             DNA += 'T'
#         elif ((DNA1[i] == 'A' and DNA2[i] == 'C') or
#               (DNA1[i] == 'C' and DNA2[i] == 'A')):
#             DNA += 'C'
#         elif ((DNA1[i] == 'A' and DNA2[i] == 'G') or
#               (DNA1[i] == 'G' and DNA2[i] == 'A')):
#             DNA += 'G'
#         elif ((DNA1[i] == 'T' and DNA2[i] == 'C') or
#               (DNA1[i] == 'C' and DNA2[i] == 'T')):
#             DNA += 'G'
#         elif ((DNA1[i] == 'T' and DNA2[i] == 'G') or
#               (DNA1[i] == 'G' and DNA2[i] == 'T')):
#             DNA += 'C'
#         elif ((DNA1[i] == 'C' and DNA2[i] == 'G') or
#               (DNA1[i] == 'G' and DNA2[i] == 'C')):
#             DNA += 'T'
#     return DNA

# def generate_keys(key1_bits):
#     seed = int(key1_bits[:8], 2) % 8
#     random.seed(seed)
#     rule1 = (seed % 8) + 1
#     rule2 = ((seed + 1) % 8) + 1
#     dna1 = DNAEncode(key1_bits, rule1)
#     reverse_key1_bits = key1_bits[::-1]
#     dna2 = DNAEncode(reverse_key1_bits, rule2)
#     xored = DNAXOR(dna1, dna2)
#     key2 = DNADecode(xored, rule=1)
    
#     rule3 = ((seed + 2) % 8) + 1
#     rule4 = ((seed + 3) % 8) + 1
#     dna3 = DNAEncode(key1_bits[-8:], rule3)
#     dna4 = DNAEncode(key2[-8:], rule4)
#     xored2 = DNAXOR(dna3, dna4)
#     key3 = DNADecode(xored2, rule=1)
    
#     return key2, key3
# # def shift_pixels(image_array, shift_value):
# #     shifted_image = np.roll(image_array, shift_value, axis=(0, 1))
# #     return shifted_image

# def shift_pixels(image_array, shift_value):
#     shifted_image = np.roll(image_array, shift_value, axis=(0, 1))
#     return shifted_image

# def generate_shift_value(key1, key2, key3):
#     """Generate an unpredictable shift value between 1-100 based on keys."""
#     combined_key = key1 + key2 + key3
#     hash_value = hashlib.sha256(combined_key.encode()).hexdigest()
#     # Convert part of hash to integer in range 1-100
#     shift_value = int(hash_value[:8], 16) % 100 + 1  # Ensures value is between 1-100
#     return shift_value

# def encryption(key1, key2, key3, image_path):
#     """Encrypt an image using triple-layer DNA encoding."""
#     image = Image.open(image_path)
#     image_array = np.array(image)

#     shift_value = generate_shift_value(key1, key2, key3)
#     image_array = shift_pixels(image_array, shift_value)
    
#     height, width, channels = image_array.shape
#     processed_image = np.zeros(shape=(height, width, channels), dtype=np.uint8)

#     index = 0
#     for i in range(height):
#         for j in range(width):
#             for k in range(channels):
#                 if index < len(key1):
#                     rule1, rule2, rule3 = (index % 8) + 1, ((index + 1) % 8) + 1, ((index + 2) % 8) + 1
#                     key_segment1 = key1[index*8:(index+1)*8]
#                     key_segment2 = key2[index*8:(index+1)*8]
#                     key_segment3 = key3[index*8:(index+1)*8]
                    
#                     original_bin = format(image_array[i, j, k], '08b')
#                     dna1 = DNAEncode(original_bin, rule1)
#                     xored1 = DNAXOR(dna1, DNAEncode(key_segment1, rule1))
#                     temp1 = DNADecode(xored1, rule1)
                    
#                     dna2 = DNAEncode(temp1, rule2)
#                     xored2 = DNAXOR(dna2, DNAEncode(key_segment2, rule2))
#                     temp2 = DNADecode(xored2, rule2)
                    
#                     dna3 = DNAEncode(temp2, rule3)
#                     xored3 = DNAXOR(dna3, DNAEncode(key_segment3, rule3))
#                     processed_image[i, j, k] = int(DNADecode(xored3, rule3), 2)
                    
#                     index += 1
#     return processed_image

# def decryption(key1, key2, key3, encrypted_image):
#     """Decrypt an image using triple-layer DNA decoding."""
#     cipher_array = np.array(encrypted_image)
    
#     height, width, channels = cipher_array.shape
#     decrypted_image = np.zeros(shape=(height, width, channels), dtype=np.uint8)

#     index = 0
#     for i in range(height):
#         for j in range(width):
#             for k in range(channels):
#                 if index < len(key1):
#                     rule1, rule2, rule3 = (index % 8) + 1, ((index + 1) % 8) + 1, ((index + 2) % 8) + 1
#                     key_segment1 = key1[index*8:(index+1)*8]
#                     key_segment2 = key2[index*8:(index+1)*8]
#                     key_segment3 = key3[index*8:(index+1)*8]
                    
#                     encrypted_bin = format(cipher_array[i, j, k], '08b')
#                     dna3 = DNAEncode(encrypted_bin, rule3)
#                     xored3 = DNAXOR(dna3, DNAEncode(key_segment3, rule3))
#                     temp3 = DNADecode(xored3, rule3)
                    
#                     dna2 = DNAEncode(temp3, rule2)
#                     xored2 = DNAXOR(dna2, DNAEncode(key_segment2, rule2))
#                     temp2 = DNADecode(xored2, rule2)
                    
#                     dna1 = DNAEncode(temp2, rule1)
#                     xored1 = DNAXOR(dna1, DNAEncode(key_segment1, rule1))
#                     decrypted_image[i, j, k] = int(DNADecode(xored1, rule1), 2)
                    
#                     index += 1
    
#     shift_value = generate_shift_value(key1, key2, key3)
#     decrypted_image = shift_pixels(decrypted_image, -shift_value)
#     return decrypted_image
# def main():
#     # Load the Lena image
#     image_path = 'khalid/lena.png' 
#     plain_image = np.array(Image.open(image_path))
    
#     # Generate keys and encrypt
#     key = "Bangladesh"
#     image_row, image_col, channel = plain_image.shape
#     target_length = image_row * image_col * channel * 8
    
#     key1_bits = ''.join(char_to_binary(c) for c in key)
#     key2_bits, key3_bits = generate_keys(key1_bits)
    
#     key1_bits = key_scramble(key1_bits, target_length)
#     key2_bits = key_scramble(key2_bits, target_length)
#     key3_bits = key_scramble(key3_bits, target_length)
    
#     cipher_image = encryption(key1_bits, key2_bits, key3_bits, image_path)
#     decrypted_image = decryption(key1_bits, key2_bits, key3_bits, cipher_image)
    
#     # Calculate difference between original and decrypted
#     difference = np.abs(plain_image.astype(int) - decrypted_image.astype(int))
#     # Create a white background image (255) and subtract differences
#     # This will make differences appear black on white background
#     difference_display = 255 - np.clip(difference * 255, 0, 255).astype(np.uint8)
    
#     # Create the figure
#     plt.figure(figsize=(12, 4))
    
#     # Subplot (a): Plain image
#     plt.subplot(1, 4, 1)
#     plt.imshow(plain_image)
#     plt.title('(a) Plain Image')
#     plt.axis('off')
    
#     # Subplot (b): Cipher image
#     plt.subplot(1, 4, 2)
#     plt.imshow(cipher_image)
#     plt.title('(b) Cipher Image')
#     plt.axis('off')
    
#     # Subplot (c): Reconstructed image
#     plt.subplot(1, 4, 3)
#     plt.imshow(decrypted_image)
#     plt.title('(c) Reconstructed Image')
#     plt.axis('off')
    
#     # Subplot (d): Difference (black on white)
#     plt.subplot(1, 4, 4)
#     plt.imshow(difference_display, cmap='gray', vmin=0, vmax=255)
#     plt.title('(d) Difference')
#     plt.axis('off')
    
#     plt.tight_layout()
#     #plt.savefig('image_reconstruction.png', bbox_inches='tight', dpi=300)
#     plt.show()

# main()




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
# def shift_pixels(image_array, shift_value):
#     shifted_image = np.roll(image_array, shift_value, axis=(0, 1))
#     return shifted_image

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
def main():
    # Load the Lena image
    image_path = 'khalid/lena.png' 
    plain_image = np.array(Image.open(image_path))
    
    # Generate keys and encrypt
    key = "Bangladesh"
    image_row, image_col, channel = plain_image.shape
    target_length = image_row * image_col * channel * 8
    
    key1_bits = ''.join(char_to_binary(c) for c in key)
    key2_bits, key3_bits = generate_keys(key1_bits)
    
    key1_bits = key_scramble(key1_bits, target_length)
    key2_bits = key_scramble(key2_bits, target_length)
    key3_bits = key_scramble(key3_bits, target_length)
    
    cipher_image = encryption(key1_bits, key2_bits, key3_bits, image_path)
    decrypted_image = decryption(key1_bits, key2_bits, key3_bits, cipher_image)
    
    # Calculate difference between original and decrypted
    difference = np.abs(plain_image.astype(int) - decrypted_image.astype(int))
    # Make the difference image black (0 where there are no differences, black where there are differences)
    difference_display = np.clip(difference, 0, 255).astype(np.uint8)
    
    # Create the figure
    plt.figure(figsize=(12, 4))
    
    # Subplot (a): Plain image
    plt.subplot(1, 4, 1)
    plt.imshow(plain_image)
    plt.title('(a) Plain Image')
    plt.axis('off')
    
    # Subplot (b): Cipher image
    plt.subplot(1, 4, 2)
    plt.imshow(cipher_image)
    plt.title('(b) Cipher Image')
    plt.axis('off')
    
    # Subplot (c): Reconstructed image
    plt.subplot(1, 4, 3)
    plt.imshow(decrypted_image)
    plt.title('(c) Reconstructed Image')
    plt.axis('off')
    
    # Subplot (d): Difference (black image)
    plt.subplot(1, 4, 4)
    plt.imshow(difference_display, cmap='gray', vmin=0, vmax=255)
    plt.title('(d) Difference')
    plt.axis('off')
    
    plt.tight_layout()
    #plt.savefig('image_reconstruction.png', bbox_inches='tight', dpi=300)
    plt.show()

main()