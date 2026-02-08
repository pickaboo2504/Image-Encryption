# import math
# import numpy as np
# from collections import Counter
# from PIL import Image
# import matplotlib.pyplot as plt
# import random
# import hashlib

# # Keeping all the original functions from the provided code
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

# # New function to add salt and pepper noise
# def add_salt_pepper_noise(image, density):
#     """Add salt and pepper noise to the image with specified density."""
#     noisy_image = np.copy(image)
    
#     # Salt noise (white pixels)
#     salt_coords = np.random.random(size=image.shape[:2]) < (density/2)
#     noisy_image[salt_coords] = 255
    
#     # Pepper noise (black pixels)
#     pepper_coords = np.random.random(size=image.shape[:2]) < (density/2)
#     noisy_image[pepper_coords] = 0
    
#     return noisy_image

# # New function to calculate PSNR
# def calculate_psnr(original, noisy):
#     """Calculate PSNR between original and noisy images."""
#     # Mean square error
#     mse = np.mean((original.astype(float) - noisy.astype(float)) ** 2)
#     if mse == 0:
#         return float('inf')  # Perfect similarity
    
#     # Maximum pixel value
#     max_pixel = np.max(original)
    
#     # PSNR calculation
#     psnr = 10 * np.log10((max_pixel ** 2) / mse)
#     return psnr

# def main():
#     # Set the image path - replace with the path to your image
#     image_path = 'khalid/Baboon512.png'  # Adjust this to your actual image path
#     image = Image.open(image_path)
#     image_array = np.array(image)
    
#     image_row, image_col, channel = image_array.shape
#     target_length = image_row * image_col * channel * 8

#     # Use the same key as in the original code
#     key = "Bangladesh"
#     key1_bits = ''.join(char_to_binary(c) for c in key)

#     key2_bits, key3_bits = generate_keys(key1_bits)

#     key1_bits = key_scramble(key1_bits, target_length)
#     key2_bits = key_scramble(key2_bits, target_length)
#     key3_bits = key_scramble(key3_bits, target_length)
    
#     # Encrypt the original image
#     cipherImage = encryption(key1_bits, key2_bits, key3_bits, image_path)
    
#     # Updated list of noise densities - added 0.002 and 0.005
#     noise_densities = [0.002, 0.005, 0.01, 0.03, 0.05]
#     psnr_values = []
    
#     # Create a figure with subplots - now we need more for all densities
#     plt.figure(figsize=(15, 10))
    
#     # Plot original image
#     plt.subplot(2, 3, 1)
#     plt.imshow(image_array)
#     plt.title("Original Image")
#     plt.axis('off')
    
#     # Process and plot for each noise density
#     for i, density in enumerate(noise_densities):
#         # Add salt and pepper noise to the encrypted image
#         noisy_cipher = add_salt_pepper_noise(cipherImage, density)
        
#         # Decrypt the noisy encrypted image
#         decrypted_noisy = decryption(key1_bits, key2_bits, key3_bits, noisy_cipher)
        
#         # Calculate PSNR between original and decrypted noisy image
#         psnr = calculate_psnr(image_array, decrypted_noisy)
#         psnr_values.append(psnr)
        
#         # Plot the decrypted image (i+2 because position 1 is the original image)
#         plt.subplot(2, 3, i+2)
#         plt.imshow(decrypted_noisy)
#         plt.title(f"Decrypted with {density} S&P noise")
#         plt.axis('off')
    
#     plt.tight_layout()
#     # plt.savefig("salt_pepper_noise_results.png")
#     plt.show()
    
#     # Print PSNR values
#     print("\nPSNR Values between original and decrypted images:")
#     for i, density in enumerate(noise_densities):
#         print(f"With {density} noise density: {psnr_values[i]:.2f} dB")
    
#     # Create a bar plot to visualize PSNR values
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(noise_densities)), psnr_values, tick_label=[str(d) for d in noise_densities])
#     plt.xlabel('Noise Density')
#     plt.ylabel('PSNR (dB)')
#     plt.title('PSNR vs. Noise Density')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     # plt.savefig("psnr_comparison.png")
#     plt.show()

# if __name__ == "__main__":
#     main()




import math
import numpy as np 
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

def char_to_binary(c):
    return format(ord(c),'08b')

def binary_to_char(n):
    return int(n, 2)

def calculate_entropy(data_list):
    length = len(data_list)
    # Count the frequency of each unique element
    freq = Counter(data_list)
    # Compute entropy
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


def encryption(ScrambleKeyBits, image_path):
    """Applies binary segments to the image's RGB channels and prints pixel values."""
    image = Image.open(image_path)
    image_array = np.array(image)
    
    #processed_image = np.copy(image)
    height, width, channels = image_array.shape
    processed_image = np.zeros(shape=(height,width,channels), dtype=np.uint8) 

    index = 0
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if index < len(ScrambleKeyBits):                    
                    rule=(index%8)+1
                    key_segment = ScrambleKeyBits[index*8:(index+1)*8]
                    original_value = image_array[i, j, k]
                    original_bin = format(original_value, '08b')

                    key_dna=DNAEncode(key_segment,rule)
                    ori_dna=DNAEncode(original_bin,rule)
                    xored_dna=DNAXOR(key_dna,ori_dna)
                    decoded_bin=DNADecode(xored_dna,rule)
                    cipher_value = int(decoded_bin, 2)

                    processed_image[i, j, k] = cipher_value
                    index += 1
                    #print(index)

    #processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

    return processed_image

def decryption(ScrambleKeyBits, processed_image):
    """Decrypts the cipher image back to the original image using the scrambled key."""
    cipher_array = np.array(processed_image)
    
    height, width, channels = cipher_array.shape
    decrypted_image = np.zeros(shape=(height, width, channels), dtype=np.uint8)

    index = 0
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if index < len(ScrambleKeyBits):
                    rule = (index % 8) + 1
                    key_segment = ScrambleKeyBits[index*8:(index+1)*8]
                    cipher_value = cipher_array[i, j, k]
                    cipher_bin = format(cipher_value, '08b')

                    key_dna = DNAEncode(key_segment, rule)
                    cipher_dna = DNAEncode(cipher_bin, rule)
                    original_dna = DNAXOR(key_dna, cipher_dna)  # Reverse XOR
                    original_bin = DNADecode(original_dna, rule)
                    original_value = int(original_bin, 2)

                    decrypted_image[i, j, k] = original_value
                    index += 1

    return decrypted_image
# New function to add salt and pepper noise
def add_salt_pepper_noise(image, density):
    """Add salt and pepper noise to the image with specified density."""
    noisy_image = np.copy(image)
    
    # Salt noise (white pixels)
    salt_coords = np.random.random(size=image.shape[:2]) < (density/2)
    noisy_image[salt_coords] = 255
    
    # Pepper noise (black pixels)
    pepper_coords = np.random.random(size=image.shape[:2]) < (density/2)
    noisy_image[pepper_coords] = 0
    
    return noisy_image

# New function to calculate PSNR
def calculate_psnr(original, noisy):
    """Calculate PSNR between original and noisy images."""
    # Mean square error
    mse = np.mean((original.astype(float) - noisy.astype(float)) ** 2)
    if mse == 0:
        return float('inf')  # Perfect similarity
    
    # Maximum pixel value
    max_pixel = np.max(original)
    
    # PSNR calculation
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def main():
    # Set the image path - replace with the path to your image
    image_path = 'khalid/Baboon512.png'  # Adjust this to your actual image path
    image = Image.open(image_path)
    image_array = np.array(image)
    
    image_row, image_col, channel = image_array.shape
    target_length = image_row * image_col * channel * 8

    # Use the same key as in the original code
    key = "Bangladesh"
    key1_bits = ''.join(char_to_binary(c) for c in key)

   

    key1_bits = key_scramble(key1_bits, target_length)
    
    
    # Encrypt the original image
    cipherImage = encryption(key1_bits, image_path)
    
    # Updated list of noise densities - added 0.002 and 0.005
    noise_densities = [0.002, 0.005, 0.01, 0.03, 0.05]
    psnr_values = []
    
    # Create a figure with subplots - now we need more for all densities
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(2, 3, 1)
    plt.imshow(image_array)
    plt.title("Original Image")
    plt.axis('off')
    
    # Process and plot for each noise density
    for i, density in enumerate(noise_densities):
        # Add salt and pepper noise to the encrypted image
        noisy_cipher = add_salt_pepper_noise(cipherImage, density)
        
        # Decrypt the noisy encrypted image
        decrypted_noisy = decryption(key1_bits, noisy_cipher)
        
        # Calculate PSNR between original and decrypted noisy image
        psnr = calculate_psnr(image_array, decrypted_noisy)
        psnr_values.append(psnr)
        
        # Plot the decrypted image (i+2 because position 1 is the original image)
        plt.subplot(2, 3, i+2)
        plt.imshow(decrypted_noisy)
        plt.title(f"Decrypted with {density} S&P noise")
        plt.axis('off')
    
    plt.tight_layout()
    # plt.savefig("salt_pepper_noise_results.png")
    plt.show()
    
    # Print PSNR values
    print("\nPSNR Values between original and decrypted images:")
    for i, density in enumerate(noise_densities):
        print(f"With {density} noise density: {psnr_values[i]:.2f} dB")
    
    # Create a bar plot to visualize PSNR values
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(noise_densities)), psnr_values, tick_label=[str(d) for d in noise_densities])
    plt.xlabel('Noise Density')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs. Noise Density')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig("psnr_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()