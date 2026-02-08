# import math
# import numpy as np
# from collections import Counter
# from PIL import Image
# import matplotlib.pyplot as plt
# import random
# import hashlib
# import io

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

# def calculate_mse(original, processed):
#     """Calculate Mean Squared Error between two images."""
#     original = np.array(original, dtype=np.float64)
#     processed = np.array(processed, dtype=np.float64)
#     mse = np.mean((original - processed) ** 2)
#     return mse

# def calculate_psnr(original, processed):
#     """Calculate Peak Signal-to-Noise Ratio between two images."""
#     mse = calculate_mse(original, processed)
#     if mse == 0:
#         return float('inf')
    
#     # Maximum pixel intensity value (for 8-bit images)
#     max_pixel = 255.0
#     psnr = 10 * math.log10((max_pixel ** 2) / mse)
#     return psnr

# def jpeg_compress(image_array, quality):
#     """Compress image using JPEG with specified quality factor."""
#     # Convert numpy array to PIL Image
#     if len(image_array.shape) == 3:
#         image = Image.fromarray(image_array.astype(np.uint8))
#     else:
#         image = Image.fromarray(image_array.astype(np.uint8), mode='L')
    
#     # Save to memory buffer with JPEG compression
#     buffer = io.BytesIO()
#     image.save(buffer, format='JPEG', quality=quality)
#     buffer.seek(0)
    
#     # Load back the compressed image
#     compressed_image = Image.open(buffer)
#     compressed_array = np.array(compressed_image)
    
#     return compressed_array

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
#     shift_value = int(hash_value[:8], 16) % 100 + 1
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
#     # Use a sample image (you can replace this with your pepper image path)
#     image_path = 'khalid/Peppers512.png'  # Replace with your pepper image path
    
#     # Create a sample image if the file doesn't exist
#     try:
#         image = Image.open(image_path)
#     except FileNotFoundError:
#         print(f"Image file {image_path} not found. Creating a sample image...")
#         # Create a sample colorful image
#         sample_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
#         sample_image = Image.fromarray(sample_image)
#         sample_image.save('sample_image.png')
#         image_path = 'sample_image.png'
#         image = Image.open(image_path)
    
#     image_array = np.array(image)
    
#     image_row, image_col, channel = image_array.shape
#     target_length = image_row * image_col * channel * 8

#     key = "Bangladesh"
#     key1_bits = ''.join(char_to_binary(c) for c in key)

#     key2_bits, key3_bits = generate_keys(key1_bits)

#     key1_bits = key_scramble(key1_bits, target_length)
#     key2_bits = key_scramble(key2_bits, target_length)
#     key3_bits = key_scramble(key3_bits, target_length)
    
#     # Encrypt the image
#     print("Encrypting image...")
#     cipherImage = encryption(key1_bits, key2_bits, key3_bits, image_path)
    
#     # JPEG compression test with quality factors
#     quality_factors = [20, 40, 60]
    
#     # Create figure for compressed encrypted images and decrypted results
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     fig.suptitle('JPEG Compression Attack Test', fontsize=16)
    
#     psnr_values = []
    
#     for i, quality in enumerate(quality_factors):
#         print(f"Testing JPEG compression with quality factor {quality}...")
        
#         # Compress the encrypted image
#         compressed_encrypted = jpeg_compress(cipherImage, quality)
        
#         # Decrypt the compressed encrypted image
#         decrypted_after_compression = decryption(key1_bits, key2_bits, key3_bits, compressed_encrypted)
        
#         # Calculate PSNR
#         psnr = calculate_psnr(image_array, decrypted_after_compression)
#         psnr_values.append(psnr)
        
#         # Display compressed encrypted image (top row)
#         axes[0, i].imshow(compressed_encrypted)
#         axes[0, i].set_title(f'Encrypted (Quality {quality})')
#         axes[0, i].axis('off')
        
#         # Display decrypted image after compression (bottom row)
#         axes[1, i].imshow(decrypted_after_compression)
#         axes[1, i].set_title(f'Decrypted (PSNR: {psnr:.2f} dB)')
#         axes[1, i].axis('off')
        
#         print(f"Quality {quality}: PSNR = {psnr:.2f} dB")
    
#     plt.tight_layout()
#     plt.show()
    
#     # Display original and normal encrypted/decrypted images
#     fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
#     # Original image
#     axes2[0].imshow(image_array)
#     axes2[0].set_title('Original Image')
#     axes2[0].axis('off')
    
#     # Encrypted image (without compression)
#     axes2[1].imshow(cipherImage)
#     axes2[1].set_title('Encrypted Image')
#     axes2[1].axis('off')
    
#     # Decrypted image (without compression)
#     decrypted_normal = decryption(key1_bits, key2_bits, key3_bits, cipherImage)
#     axes2[2].imshow(decrypted_normal)
#     axes2[2].set_title('Decrypted Image (No Compression)')
#     axes2[2].axis('off')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print summary
#     print("\n=== JPEG Compression Attack Results ===")
#     for i, (quality, psnr) in enumerate(zip(quality_factors, psnr_values)):
#         print(f"Quality Factor {quality}: PSNR = {psnr:.2f} dB")
    
#     # Calculate and display entropy values
#     original_1d = image_array.flatten()
#     cipher_1d = cipherImage.flatten()
    
#     print(f"\nOriginal Image Entropy: {calculate_entropy(original_1d):.4f}")
#     print(f"Encrypted Image Entropy: {calculate_entropy(cipher_1d):.4f}")
    
#     # Display histograms
#     fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    
#     axes3[0].hist(original_1d, bins=256, alpha=0.7, color='blue')
#     axes3[0].set_title('Original Image Histogram')
#     axes3[0].set_xlabel('Pixel Intensity')
#     axes3[0].set_ylabel('Frequency')
    
#     axes3[1].hist(cipher_1d, bins=256, alpha=0.7, color='red')
#     axes3[1].set_title('Encrypted Image Histogram')
#     axes3[1].set_xlabel('Pixel Intensity')
#     axes3[1].set_ylabel('Frequency')
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()


import math
import numpy as np
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import random
import hashlib
import io

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

def calculate_mse(original, processed):
    """Calculate Mean Squared Error between two images."""
    original = np.array(original, dtype=np.float64)
    processed = np.array(processed, dtype=np.float64)
    mse = np.mean((original - processed) ** 2)
    return mse

def calculate_psnr(original, processed):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = calculate_mse(original, processed)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 10 * math.log10((max_pixel ** 2) / mse)
    return psnr

def jpeg_compress(image_array, quality):
    """Compress image using JPEG with specified quality factor."""
    if len(image_array.shape) == 3:
        image = Image.fromarray(image_array.astype(np.uint8))
    else:
        image = Image.fromarray(image_array.astype(np.uint8), mode='L')
    
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return np.array(compressed_image)

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
    return np.roll(image_array, shift_value, axis=(0, 1))

def generate_shift_value(key1, key2, key3):
    combined_key = key1 + key2 + key3
    hash_value = hashlib.sha256(combined_key.encode()).hexdigest()
    return int(hash_value[:8], 16) % 100 + 1

def apply_median_filter(image_array, window_size=3):
    """Custom median filter implementation without OpenCV"""
    pad = window_size // 2
    filtered_image = np.zeros_like(image_array)
    
    if len(image_array.shape) == 3:
        # Color image
        for c in range(image_array.shape[2]):
            padded = np.pad(image_array[:, :, c], pad, mode='reflect')
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    neighborhood = padded[i:i+window_size, j:j+window_size]
                    filtered_image[i, j, c] = np.median(neighborhood)
    else:
        # Grayscale image
        padded = np.pad(image_array, pad, mode='reflect')
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                neighborhood = padded[i:i+window_size, j:j+window_size]
                filtered_image[i, j] = np.median(neighborhood)
    
    return filtered_image.astype(np.uint8)

def encryption(key1, key2, key3, image_path):
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
    
    # Apply our custom median filter
    decrypted_image = apply_median_filter(decrypted_image)
    
    return decrypted_image

# [Previous imports and function definitions remain exactly the same until the main() function]

def main():
    image_path = 'khalid/lena.png'
    
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Image file {image_path} not found. Creating a sample image...")
        sample_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        sample_image = Image.fromarray(sample_image)
        sample_image.save('sample_image.png')
        image_path = 'sample_image.png'
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
    
    print("Encrypting image...")
    cipherImage = encryption(key1_bits, key2_bits, key3_bits, image_path)
    
    quality_factors = [20, 40, 60]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Remove the main title
    # fig.suptitle('JPEG Compression Attack Test (Improved)', fontsize=16)
    
    psnr_values = []
    
    for i, quality in enumerate(quality_factors):
        print(f"Testing JPEG compression with quality factor {quality}...")
        
        compressed_encrypted = jpeg_compress(cipherImage, quality)
        decrypted_after_compression = decryption(key1_bits, key2_bits, key3_bits, compressed_encrypted)
        
        psnr = calculate_psnr(image_array, decrypted_after_compression)
        psnr_values.append(psnr)
        
        # Top row (a, b, c)
        axes[0, i].imshow(compressed_encrypted)
        axes[0, i].set_title('(' + chr(97 + i) + ')', fontsize=14)  # 97 is ASCII for 'a'
        axes[0, i].axis('off')
        
        # Bottom row (d, e, f)
        axes[1, i].imshow(decrypted_after_compression)
        axes[1, i].set_title('(' + chr(100 + i) + ')', fontsize=14)  # 100 is ASCII for 'd'
        axes[1, i].axis('off')
        
        print(f"Quality {quality}: PSNR = {psnr:.2f} dB")
    
    plt.tight_layout()
    output_filename = 'jpeg_compression_attack_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {output_filename}")
    plt.show()
    
    # [Rest of the code remains the same]

if __name__ == "__main__":
    main()