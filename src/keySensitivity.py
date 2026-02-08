# # import math
# # import numpy as np
# # from collections import Counter
# # from PIL import Image
# # import matplotlib.pyplot as plt
# # import random
# # import hashlib

# # def char_to_binary(c):
# #     """Convert a character to its 8-bit binary representation."""
# #     return format(ord(c), '08b')

# # def binary_to_char(n):
# #     """Convert an 8-bit binary string to its character representation."""
# #     return chr(int(n, 2))

# # def calculate_entropy(data_list):
# #     """Calculate the entropy of a given data list."""
# #     length = len(data_list)
# #     freq = Counter(data_list)
# #     entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
# #     return entropy

# # def key_scramble(key, target_length):
# #     """Expand a key to the target length using scrambling operations."""
# #     large_key = key
# #     reverse_temp = key[::-1]
# #     large_key += reverse_temp

# #     while len(large_key) < target_length:
# #         temp = list(key)
# #         for i in range(len(temp) - 1):
# #             if temp[i] != temp[i + 1]:
# #                 temp[i], temp[i + 1] = temp[i + 1], temp[i]
# #                 temp_str = ''.join(temp)
# #                 for j in range(len(temp_str)):
# #                     large_key += temp_str
# #                     large_key += temp_str[::-1]
# #                     temp_str = temp_str[1:] + temp_str[0]
# #                     if len(large_key) >= target_length:
# #                         break
# #             if len(large_key) >= target_length:
# #                 break
# #         key = large_key[:target_length]

# #     return large_key[:target_length]

# # def DNADecode(DNA, rule):
# #     """Decode DNA sequence to binary based on the specified rule."""
# #     Bits = ''
# #     if rule == 1:
# #         for i in range(4):
# #             if DNA[i] == 'A': Bits += '00'
# #             elif DNA[i] == 'G': Bits += '01'
# #             elif DNA[i] == 'C': Bits += '10'
# #             elif DNA[i] == 'T': Bits += '11'
# #     elif rule == 2:
# #         for i in range(4):
# #             if DNA[i] == 'A': Bits += '00'
# #             elif DNA[i] == 'C': Bits += '01'
# #             elif DNA[i] == 'G': Bits += '10'
# #             elif DNA[i] == 'T': Bits += '11'
# #     elif rule == 3:
# #         for i in range(4):
# #             if DNA[i] == 'T': Bits += '00'
# #             elif DNA[i] == 'G': Bits += '01'
# #             elif DNA[i] == 'C': Bits += '10'
# #             elif DNA[i] == 'A': Bits += '11'
# #     elif rule == 4:
# #         for i in range(4):
# #             if DNA[i] == 'T': Bits += '00'
# #             elif DNA[i] == 'C': Bits += '01'
# #             elif DNA[i] == 'G': Bits += '10'
# #             elif DNA[i] == 'A': Bits += '11'
# #     elif rule == 5:
# #         for i in range(4):
# #             if DNA[i] == 'C': Bits += '00'
# #             elif DNA[i] == 'T': Bits += '01'
# #             elif DNA[i] == 'A': Bits += '10'
# #             elif DNA[i] == 'G': Bits += '11'
# #     elif rule == 6:
# #         for i in range(4):
# #             if DNA[i] == 'C': Bits += '00'
# #             elif DNA[i] == 'A': Bits += '01'
# #             elif DNA[i] == 'T': Bits += '10'
# #             elif DNA[i] == 'G': Bits += '11'
# #     elif rule == 7:
# #         for i in range(4):
# #             if DNA[i] == 'G': Bits += '00'
# #             elif DNA[i] == 'T': Bits += '01'
# #             elif DNA[i] == 'A': Bits += '10'
# #             elif DNA[i] == 'C': Bits += '11'
# #     elif rule == 8:
# #         for i in range(4):
# #             if DNA[i] == 'G': Bits += '00'
# #             elif DNA[i] == 'A': Bits += '01'
# #             elif DNA[i] == 'T': Bits += '10'
# #             elif DNA[i] == 'C': Bits += '11'
# #     return Bits

# # def DNAEncode(bits, rule):
# #     """Encode binary to DNA sequence based on the specified rule."""
# #     DNA = ''
# #     if rule == 1:
# #         for i in range(0, 8, 2):
# #             if bits[i:i+2] == '00': DNA += 'A'
# #             elif bits[i:i+2] == '01': DNA += 'G'
# #             elif bits[i:i+2] == '10': DNA += 'C'
# #             elif bits[i:i+2] == '11': DNA += 'T'
# #     elif rule == 2:
# #         for i in range(0, 8, 2):
# #             if bits[i:i+2] == '00': DNA += 'A'
# #             elif bits[i:i+2] == '01': DNA += 'C'
# #             elif bits[i:i+2] == '10': DNA += 'G'
# #             elif bits[i:i+2] == '11': DNA += 'T'
# #     elif rule == 3:
# #         for i in range(0, 8, 2):
# #             if bits[i:i+2] == '00': DNA += 'T'
# #             elif bits[i:i+2] == '01': DNA += 'G'
# #             elif bits[i:i+2] == '10': DNA += 'C'
# #             elif bits[i:i+2] == '11': DNA += 'A'
# #     elif rule == 4:
# #         for i in range(0, 8, 2):
# #             if bits[i:i+2] == '00': DNA += 'T'
# #             elif bits[i:i+2] == '01': DNA += 'C'
# #             elif bits[i:i+2] == '10': DNA += 'G'
# #             elif bits[i:i+2] == '11': DNA += 'A'
# #     elif rule == 5:
# #         for i in range(0, 8, 2):
# #             if bits[i:i+2] == '00': DNA += 'C'
# #             elif bits[i:i+2] == '01': DNA += 'T'
# #             elif bits[i:i+2] == '10': DNA += 'A'
# #             elif bits[i:i+2] == '11': DNA += 'G'
# #     elif rule == 6:
# #         for i in range(0, 8, 2):
# #             if bits[i:i+2] == '00': DNA += 'C'
# #             elif bits[i:i+2] == '01': DNA += 'A'
# #             elif bits[i:i+2] == '10': DNA += 'T'
# #             elif bits[i:i+2] == '11': DNA += 'G'
# #     elif rule == 7:
# #         for i in range(0, 8, 2):
# #             if bits[i:i+2] == '00': DNA += 'G'
# #             elif bits[i:i+2] == '01': DNA += 'T'
# #             elif bits[i:i+2] == '10': DNA += 'A'
# #             elif bits[i:i+2] == '11': DNA += 'C'
# #     elif rule == 8:
# #         for i in range(0, 8, 2):
# #             if bits[i:i+2] == '00': DNA += 'G'
# #             elif bits[i:i+2] == '01': DNA += 'A'
# #             elif bits[i:i+2] == '10': DNA += 'T'
# #             elif bits[i:i+2] == '11': DNA += 'C'
# #     return DNA

# # def DNAXOR(DNA1, DNA2):
# #     """Perform XOR operation on two DNA sequences."""
# #     DNA = ''
# #     for i in range(4):
# #         if ((DNA1[i] == 'A' and DNA2[i] == 'A') or
# #             (DNA1[i] == 'T' and DNA2[i] == 'T') or
# #             (DNA1[i] == 'C' and DNA2[i] == 'C') or
# #             (DNA1[i] == 'G' and DNA2[i] == 'G')):
# #             DNA += 'A'
# #         elif ((DNA1[i] == 'A' and DNA2[i] == 'T') or
# #               (DNA1[i] == 'T' and DNA2[i] == 'A')):
# #             DNA += 'T'
# #         elif ((DNA1[i] == 'A' and DNA2[i] == 'C') or
# #               (DNA1[i] == 'C' and DNA2[i] == 'A')):
# #             DNA += 'C'
# #         elif ((DNA1[i] == 'A' and DNA2[i] == 'G') or
# #               (DNA1[i] == 'G' and DNA2[i] == 'A')):
# #             DNA += 'G'
# #         elif ((DNA1[i] == 'T' and DNA2[i] == 'C') or
# #               (DNA1[i] == 'C' and DNA2[i] == 'T')):
# #             DNA += 'G'
# #         elif ((DNA1[i] == 'T' and DNA2[i] == 'G') or
# #               (DNA1[i] == 'G' and DNA2[i] == 'T')):
# #             DNA += 'C'
# #         elif ((DNA1[i] == 'C' and DNA2[i] == 'G') or
# #               (DNA1[i] == 'G' and DNA2[i] == 'C')):
# #             DNA += 'T'
# #     return DNA

# # def generate_keys(key1_bits):
# #     seed = int(key1_bits[:8], 2) % 8
# #     random.seed(seed)
# #     rule1 = (seed % 8) + 1
# #     rule2 = ((seed + 1) % 8) + 1
# #     dna1 = DNAEncode(key1_bits, rule1)
# #     reverse_key1_bits = key1_bits[::-1]
# #     dna2 = DNAEncode(reverse_key1_bits, rule2)
# #     xored = DNAXOR(dna1, dna2)
# #     key2 = DNADecode(xored, rule=1)
    
# #     rule3 = ((seed + 2) % 8) + 1
# #     rule4 = ((seed + 3) % 8) + 1
# #     dna3 = DNAEncode(key1_bits[-8:], rule3)
# #     dna4 = DNAEncode(key2[-8:], rule4)
# #     xored2 = DNAXOR(dna3, dna4)
# #     key3 = DNADecode(xored2, rule=1)
    
# #     return key2, key3

# # def shift_pixels(image_array, shift_value):
# #     shifted_image = np.roll(image_array, shift_value, axis=(0, 1))
# #     return shifted_image

# # def generate_shift_value(key1, key2, key3):
# #     """Generate an unpredictable shift value between 1-100 based on keys."""
# #     combined_key = key1 + key2 + key3
# #     hash_value = hashlib.sha256(combined_key.encode()).hexdigest()
# #     shift_value = int(hash_value[:8], 16) % 100 + 1  # Ensures value is between 1-100
# #     return shift_value

# # def encryption(key1, key2, key3, image_path):
# #     """Encrypt an image using triple-layer DNA encoding."""
# #     if isinstance(image_path, str):
# #         image = Image.open(image_path)
# #         image_array = np.array(image)
# #     else:
# #         image_array = image_path  # If already a numpy array

# #     shift_value = generate_shift_value(key1, key2, key3)
# #     image_array = shift_pixels(image_array, shift_value)
    
# #     height, width, channels = image_array.shape
# #     processed_image = np.zeros(shape=(height, width, channels), dtype=np.uint8)

# #     index = 0
# #     for i in range(height):
# #         for j in range(width):
# #             for k in range(channels):
# #                 if index < len(key1):
# #                     rule1, rule2, rule3 = (index % 8) + 1, ((index + 1) % 8) + 1, ((index + 2) % 8) + 1
# #                     key_segment1 = key1[index*8:(index+1)*8]
# #                     key_segment2 = key2[index*8:(index+1)*8]
# #                     key_segment3 = key3[index*8:(index+1)*8]
                    
# #                     original_bin = format(image_array[i, j, k], '08b')
# #                     dna1 = DNAEncode(original_bin, rule1)
# #                     xored1 = DNAXOR(dna1, DNAEncode(key_segment1, rule1))
# #                     temp1 = DNADecode(xored1, rule1)
                    
# #                     dna2 = DNAEncode(temp1, rule2)
# #                     xored2 = DNAXOR(dna2, DNAEncode(key_segment2, rule2))
# #                     temp2 = DNADecode(xored2, rule2)
                    
# #                     dna3 = DNAEncode(temp2, rule3)
# #                     xored3 = DNAXOR(dna3, DNAEncode(key_segment3, rule3))
# #                     processed_image[i, j, k] = int(DNADecode(xored3, rule3), 2)
                    
# #                     index += 1
# #     return processed_image

# # def decryption(key1, key2, key3, encrypted_image):
# #     """Decrypt an image using triple-layer DNA decoding."""
# #     cipher_array = np.array(encrypted_image)
    
# #     height, width, channels = cipher_array.shape
# #     decrypted_image = np.zeros(shape=(height, width, channels), dtype=np.uint8)

# #     index = 0
# #     for i in range(height):
# #         for j in range(width):
# #             for k in range(channels):
# #                 if index < len(key1):
# #                     rule1, rule2, rule3 = (index % 8) + 1, ((index + 1) % 8) + 1, ((index + 2) % 8) + 1
# #                     key_segment1 = key1[index*8:(index+1)*8]
# #                     key_segment2 = key2[index*8:(index+1)*8]
# #                     key_segment3 = key3[index*8:(index+1)*8]
                    
# #                     encrypted_bin = format(cipher_array[i, j, k], '08b')
# #                     dna3 = DNAEncode(encrypted_bin, rule3)
# #                     xored3 = DNAXOR(dna3, DNAEncode(key_segment3, rule3))
# #                     temp3 = DNADecode(xored3, rule3)
                    
# #                     dna2 = DNAEncode(temp3, rule2)
# #                     xored2 = DNAXOR(dna2, DNAEncode(key_segment2, rule2))
# #                     temp2 = DNADecode(xored2, rule2)
                    
# #                     dna1 = DNAEncode(temp2, rule1)
# #                     xored1 = DNAXOR(dna1, DNAEncode(key_segment1, rule1))
# #                     decrypted_image[i, j, k] = int(DNADecode(xored1, rule1), 2)
                    
# #                     index += 1
    
# #     shift_value = generate_shift_value(key1, key2, key3)
# #     decrypted_image = shift_pixels(decrypted_image, -shift_value)
# #     return decrypted_image

# # # Implement NPCR (Number of Pixel Change Rate) as per equation (15) and (16)
# # def calculate_NPCR(image1, image2):
# #     """
# #     Calculate NPCR (Number of Pixels Change Rate) between two images.
# #     NPCR measures the percentage of different pixels between two images.
    
# #     Formula: NPCR = (Sum of G1(i,j)) / (h * w) * 100
# #     where G1(i,j) = 0 if C1(i,j) == C2(i,j), 1 otherwise
# #     """
# #     if image1.shape != image2.shape:
# #         raise ValueError("Images must have the same dimensions")
    
# #     height, width, channels = image1.shape
# #     total_pixels = height * width * channels
# #     different_pixels = 0
    
# #     for i in range(height):
# #         for j in range(width):
# #             for k in range(channels):
# #                 if image1[i, j, k] != image2[i, j, k]:
# #                     different_pixels += 1
    
# #     npcr = (different_pixels / total_pixels) * 100
# #     return npcr

# # # Implement UACI (Unified Average Changing Intensity) as per equation (17)
# # def calculate_UACI(image1, image2):
# #     """
# #     Calculate UACI (Unified Average Changing Intensity) between two images.
# #     UACI measures the average intensity difference between two images.
    
# #     Formula: UACI = (1 / (255 * h * w)) * Sum(|C1(i,j) - C2(i,j)|) * 100
# #     """
# #     if image1.shape != image2.shape:
# #         raise ValueError("Images must have the same dimensions")
    
# #     height, width, channels = image1.shape
# #     total_pixels = height * width * channels
# #     intensity_diff_sum = 0
    
# #     for i in range(height):
# #         for j in range(width):
# #             for k in range(channels):
# #                 intensity_diff_sum += abs(int(image1[i, j, k]) - int(image2[i, j, k]))
    
# #     uaci = (intensity_diff_sum / (255 * total_pixels)) * 100
# #     return uaci

# # def key_sensitivity_analysis(image_path):
# #     """
# #     Perform key sensitivity analysis as described in the paper.
# #     Tests both encryption sensitivity and decryption sensitivity.
# #     """
# #     print("Performing Key Sensitivity Analysis...")
    
# #     # Load the image
# #     image = Image.open(image_path)
# #     image_array = np.array(image)
    
# #     image_row, image_col, channel = image_array.shape
# #     target_length = image_row * image_col * channel * 8

# #     # Generate first key (K1)
# #     original_key = "Bangladesh"
# #     key1_bits_original = ''.join(char_to_binary(c) for c in original_key)
    
# #     # Generate second key (K2) with a small change
# #     # In this case, we'll modify the last character slightly
# #     modified_key = "Bangladesg"  # 'h' changed to 'g'
# #     key1_bits_modified = ''.join(char_to_binary(c) for c in modified_key)
    
# #     # Generate all necessary keys for both original and modified
# #     key2_bits_original, key3_bits_original = generate_keys(key1_bits_original)
# #     key2_bits_modified, key3_bits_modified = generate_keys(key1_bits_modified)
    
# #     # Expand keys to required length
# #     key1_bits_original = key_scramble(key1_bits_original, target_length)
# #     key2_bits_original = key_scramble(key2_bits_original, target_length)
# #     key3_bits_original = key_scramble(key3_bits_original, target_length)
    
# #     key1_bits_modified = key_scramble(key1_bits_modified, target_length)
# #     key2_bits_modified = key_scramble(key2_bits_modified, target_length)
# #     key3_bits_modified = key_scramble(key3_bits_modified, target_length)
    
# #     # Test I: Encryption Sensitivity
# #     print("\n--- I. Encryption Sensitivity Test ---")
# #     print("Encrypting the same image with two slightly different keys...")
    
# #     # Encrypt with K1
# #     encrypted_image1 = encryption(key1_bits_original, key2_bits_original, key3_bits_original, image_path)
    
# #     # Encrypt with K2
# #     encrypted_image2 = encryption(key1_bits_modified, key2_bits_modified, key3_bits_modified, image_path)
    
# #     # Calculate NPCR and UACI between the two encrypted images
# #     npcr_encryption = calculate_NPCR(encrypted_image1, encrypted_image2)
# #     uaci_encryption = calculate_UACI(encrypted_image1, encrypted_image2)
    
# #     print(f"NPCR between encrypted images: {npcr_encryption:.4f}%")
# #     print(f"UACI between encrypted images: {uaci_encryption:.4f}%")
    
# #     # Ideal values for reference
# #     print("Ideal values - NPCR: 99.6094%, UACI: 33.4635%")
    
# #     # Test II: Decryption Sensitivity
# #     print("\n--- II. Decryption Sensitivity Test ---")
# #     print("Encrypting with key K1 and attempting to decrypt with key K2...")
    
# #     # Decrypt with wrong key
# #     decrypted_with_wrong_key = decryption(key1_bits_modified, key2_bits_modified, key3_bits_modified, encrypted_image1)
    
# #     # Decrypt with correct key for comparison
# #     decrypted_with_correct_key = decryption(key1_bits_original, key2_bits_original, key3_bits_original, encrypted_image1)
    
# #     # Calculate difference between original image and incorrectly decrypted image
# #     npcr_decryption = calculate_NPCR(image_array, decrypted_with_wrong_key)
# #     uaci_decryption = calculate_UACI(image_array, decrypted_with_wrong_key)
    
# #     print(f"NPCR between original and incorrectly decrypted: {npcr_decryption:.4f}%")
# #     print(f"UACI between original and incorrectly decrypted: {uaci_decryption:.4f}%")
    
# #     # Check if correctly decrypted image matches original
# #     npcr_correct = calculate_NPCR(image_array, decrypted_with_correct_key)
# #     print(f"NPCR between original and correctly decrypted: {npcr_correct:.4f}%")
    
# #     # Visualize results
# #     plt.figure(figsize=(15, 10))
    
# #     plt.subplot(2, 3, 1)
# #     plt.imshow(image_array)
# #     plt.title("Original Image")
    
# #     plt.subplot(2, 3, 2)
# #     plt.imshow(encrypted_image1)
# #     plt.title("Encrypted with Key K1")
    
# #     plt.subplot(2, 3, 3)
# #     plt.imshow(encrypted_image2)
# #     plt.title("Encrypted with Key K2")
    
# #     plt.subplot(2, 3, 4)
# #     plt.imshow(decrypted_with_correct_key)
# #     plt.title("Decrypted with Correct Key")
    
# #     plt.subplot(2, 3, 5)
# #     plt.imshow(decrypted_with_wrong_key)
# #     plt.title("Decrypted with Wrong Key")
    
# #     # Calculate and display histograms
# #     plt.figure(figsize=(15, 5))
    
# #     plt.subplot(1, 3, 1)
# #     plt.hist(image_array.flatten(), bins=256, color='blue', alpha=0.7)
# #     plt.title("Original Image Histogram")
    
# #     plt.subplot(1, 3, 2)
# #     plt.hist(encrypted_image1.flatten(), bins=256, color='red', alpha=0.7)
# #     plt.title("Encrypted Image (K1) Histogram")
    
# #     plt.subplot(1, 3, 3)
# #     plt.hist(encrypted_image2.flatten(), bins=256, color='green', alpha=0.7)
# #     plt.title("Encrypted Image (K2) Histogram")
    
# #     plt.tight_layout()
# #     plt.show()
    
# #     # Summary of results
# #     print("\n--- Key Sensitivity Analysis Summary ---")
# #     print(f"1. Encryption Sensitivity: NPCR = {npcr_encryption:.4f}%, UACI = {uaci_encryption:.4f}%")
# #     print(f"2. Decryption Sensitivity: NPCR = {npcr_decryption:.4f}%, UACI = {uaci_decryption:.4f}%")
    
# #     if npcr_encryption > 99.5 and 33 < uaci_encryption < 34:
# #         print("✓ The algorithm shows excellent encryption sensitivity to key changes.")
# #     else:
# #         print("⚠ The encryption sensitivity may need improvement.")
        
# #     if npcr_decryption > 99.5:
# #         print("✓ The algorithm shows excellent decryption sensitivity to key changes.")
# #     else:
# #         print("⚠ The decryption sensitivity may need improvement.")
    
# #     return {
# #         'npcr_encryption': npcr_encryption,
# #         'uaci_encryption': uaci_encryption,
# #         'npcr_decryption': npcr_decryption,
# #         'uaci_decryption': uaci_decryption
# #     }

# # if __name__ == "__main__":
# #     # You can specify the path to your image here
# #     image_path = 'khalid/Lena256.png'  # Replace with your actual image path
    
# #     # Run the key sensitivity analysis
# #     results = key_sensitivity_analysis(image_path)
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

# def shift_pixels(image_array, shift_value):
#     shifted_image = np.roll(image_array, shift_value, axis=(0, 1))
#     return shifted_image

# def generate_shift_value(key1, key2, key3):
#     """Generate an unpredictable shift value between 1-100 based on keys."""
#     combined_key = key1 + key2 + key3
#     hash_value = hashlib.sha256(combined_key.encode()).hexdigest()
#     shift_value = int(hash_value[:8], 16) % 100 + 1  # Ensures value is between 1-100
#     return shift_value

# def encryption(key1, key2, key3, image_path):
#     """Encrypt an image using triple-layer DNA encoding."""
#     if isinstance(image_path, str):
#         image = Image.open(image_path)
#         image_array = np.array(image)
#     else:
#         image_array = image_path  # If already a numpy array

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

# # Implement NPCR (Number of Pixel Change Rate) as per equation (15) and (16)
# def calculate_NPCR(image1, image2):
#     """
#     Calculate NPCR (Number of Pixels Change Rate) between two images.
#     NPCR measures the percentage of different pixels between two images.
    
#     Formula: NPCR = (Sum of G1(i,j)) / (h * w) * 100
#     where G1(i,j) = 0 if C1(i,j) == C2(i,j), 1 otherwise
#     """
#     if image1.shape != image2.shape:
#         raise ValueError("Images must have the same dimensions")
    
#     height, width, channels = image1.shape
#     total_pixels = height * width * channels
#     different_pixels = 0
    
#     for i in range(height):
#         for j in range(width):
#             for k in range(channels):
#                 if image1[i, j, k] != image2[i, j, k]:
#                     different_pixels += 1
    
#     npcr = (different_pixels / total_pixels) * 100
#     return npcr

# # Implement UACI (Unified Average Changing Intensity) as per equation (17)
# def calculate_UACI(image1, image2):
#     """
#     Calculate UACI (Unified Average Changing Intensity) between two images.
#     UACI measures the average intensity difference between two images.
    
#     Formula: UACI = (1 / (255 * h * w)) * Sum(|C1(i,j) - C2(i,j)|) * 100
#     """
#     if image1.shape != image2.shape:
#         raise ValueError("Images must have the same dimensions")
    
#     height, width, channels = image1.shape
#     total_pixels = height * width * channels
#     intensity_diff_sum = 0
    
#     for i in range(height):
#         for j in range(width):
#             for k in range(channels):
#                 intensity_diff_sum += abs(int(image1[i, j, k]) - int(image2[i, j, k]))
    
#     uaci = (intensity_diff_sum / (255 * total_pixels)) * 100
#     return uaci

# def key_sensitivity_analysis(image_path):
#     """
#     Perform key sensitivity analysis as described in the paper.
#     Tests both encryption sensitivity and decryption sensitivity.
#     """
#     print("Performing Key Sensitivity Analysis...")
    
#     # Load the image
#     image = Image.open(image_path)
#     image_array = np.array(image)
    
#     image_row, image_col, channel = image_array.shape
#     target_length = image_row * image_col * channel * 8

#     # Generate first key (K1)
#     original_key = "Bangladesh"
#     key1_bits_original = ''.join(char_to_binary(c) for c in original_key)
    
#     # Generate second key (K2) with a small change
#     # In this case, we'll modify the last character slightly
#     modified_key = "Bangladesg"  # 'h' changed to 'g'
#     key1_bits_modified = ''.join(char_to_binary(c) for c in modified_key)
    
#     # Generate all necessary keys for both original and modified
#     key2_bits_original, key3_bits_original = generate_keys(key1_bits_original)
#     key2_bits_modified, key3_bits_modified = generate_keys(key1_bits_modified)
    
#     # Expand keys to required length
#     key1_bits_original = key_scramble(key1_bits_original, target_length)
#     key2_bits_original = key_scramble(key2_bits_original, target_length)
#     key3_bits_original = key_scramble(key3_bits_original, target_length)
    
#     key1_bits_modified = key_scramble(key1_bits_modified, target_length)
#     key2_bits_modified = key_scramble(key2_bits_modified, target_length)
#     key3_bits_modified = key_scramble(key3_bits_modified, target_length)
    
#     # Test I: Encryption Sensitivity Test
#     print("\n--- I. Encryption Sensitivity Test ---")
#     print("Encrypting the same image with two slightly different keys...")
    
#     # Encrypt with K1
#     encrypted_image1 = encryption(key1_bits_original, key2_bits_original, key3_bits_original, image_path)
    
#     # Encrypt with K2
#     encrypted_image2 = encryption(key1_bits_modified, key2_bits_modified, key3_bits_modified, image_path)
    
#     # Calculate NPCR and UACI between the two encrypted images
#     npcr_encryption = calculate_NPCR(encrypted_image1, encrypted_image2)
#     uaci_encryption = calculate_UACI(encrypted_image1, encrypted_image2)
    
#     print(f"NPCR between encrypted images: {npcr_encryption:.4f}%")
#     print(f"UACI between encrypted images: {uaci_encryption:.4f}%")
    
#     # Ideal values for reference
#     print("Ideal values - NPCR: 99.6094%, UACI: 33.4635%")
    
#     # Test II: Decryption Sensitivity
#     print("\n--- II. Decryption Sensitivity Test ---")
#     print("Encrypting with key K1 and attempting to decrypt with key K2...")
    
#     # Decrypt with wrong key
#     decrypted_with_wrong_key = decryption(key1_bits_modified, key2_bits_modified, key3_bits_modified, encrypted_image1)
    
#     # Decrypt with correct key for comparison
#     decrypted_with_correct_key = decryption(key1_bits_original, key2_bits_original, key3_bits_original, encrypted_image1)
    
#     # Calculate difference between original image and incorrectly decrypted image
#     npcr_decryption = calculate_NPCR(image_array, decrypted_with_wrong_key)
#     uaci_decryption = calculate_UACI(image_array, decrypted_with_wrong_key)
    
#     print(f"NPCR between original and incorrectly decrypted: {npcr_decryption:.4f}%")
#     print(f"UACI between original and incorrectly decrypted: {uaci_decryption:.4f}%")
    
#     # Check if correctly decrypted image matches original
#     npcr_correct = calculate_NPCR(image_array, decrypted_with_correct_key)
#     print(f"NPCR between original and correctly decrypted: {npcr_correct:.4f}%")
    
#     # Visualize results - MODIFIED TO ONLY SHOW PLAIN IMAGE AND DECRYPTED IMAGES
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 3, 1)
#     plt.imshow(image_array)
#     plt.title("Original Image")
    
#     plt.subplot(1, 3, 2)
#     plt.imshow(decrypted_with_correct_key)
#     plt.title("Decrypted with Correct Key")
    
#     plt.subplot(1, 3, 3)
#     plt.imshow(decrypted_with_wrong_key)
#     plt.title("Decrypted with Wrong Key")
    
#     plt.tight_layout()
#     # plt.savefig("keysensetivity.png")
#     plt.show()
    
#     # Just show histograms for original image
#     plt.figure(figsize=(8, 5))
#     plt.hist(image_array.flatten(), bins=256, color='blue', alpha=0.7)
#     plt.title("Original Image Histogram")
#     plt.tight_layout()
#     plt.show()
    
#     # Summary of results
#     print("\n--- Key Sensitivity Analysis Summary ---")
#     print(f"1. Encryption Sensitivity: NPCR = {npcr_encryption:.4f}%, UACI = {uaci_encryption:.4f}%")
#     print(f"2. Decryption Sensitivity: NPCR = {npcr_decryption:.4f}%, UACI = {uaci_decryption:.4f}%")
    
#     if npcr_encryption > 99.5 and 33 < uaci_encryption < 34:
#         print("✓ The algorithm shows excellent encryption sensitivity to key changes.")
#     else:
#         print("⚠ The encryption sensitivity may need improvement.")
        
#     if npcr_decryption > 99.5:
#         print("✓ The algorithm shows excellent decryption sensitivity to key changes.")
#     else:
#         print("⚠ The decryption sensitivity may need improvement.")
    
#     return {
#         'npcr_encryption': npcr_encryption,
#         'uaci_encryption': uaci_encryption,
#         'npcr_decryption': npcr_decryption,
#         'uaci_decryption': uaci_decryption
#     }

# if __name__ == "__main__":
#     # You can specify the path to your image here
#     image_path = 'khalid/Lena256.png'  # Replace with your actual image path
    
#     # Run the key sensitivity analysis
#     results = key_sensitivity_analysis(image_path)



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
# Implement NPCR (Number of Pixel Change Rate) as per equation (15) and (16)
def calculate_NPCR(image1, image2):
    """
    Calculate NPCR (Number of Pixels Change Rate) between two images.
    NPCR measures the percentage of different pixels between two images.
    
    Formula: NPCR = (Sum of G1(i,j)) / (h * w) * 100
    where G1(i,j) = 0 if C1(i,j) == C2(i,j), 1 otherwise
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    height, width, channels = image1.shape
    total_pixels = height * width * channels
    different_pixels = 0
    
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if image1[i, j, k] != image2[i, j, k]:
                    different_pixels += 1
    
    npcr = (different_pixels / total_pixels) * 100
    return npcr

# Implement UACI (Unified Average Changing Intensity) as per equation (17)
def calculate_UACI(image1, image2):
    """
    Calculate UACI (Unified Average Changing Intensity) between two images.
    UACI measures the average intensity difference between two images.
    
    Formula: UACI = (1 / (255 * h * w)) * Sum(|C1(i,j) - C2(i,j)|) * 100
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    height, width, channels = image1.shape
    total_pixels = height * width * channels
    intensity_diff_sum = 0
    
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                intensity_diff_sum += abs(int(image1[i, j, k]) - int(image2[i, j, k]))
    
    uaci = (intensity_diff_sum / (255 * total_pixels)) * 100
    return uaci

def key_sensitivity_analysis(image_path):
    """
    Perform key sensitivity analysis as described in the paper.
    Tests both encryption sensitivity and decryption sensitivity.
    """
    print("Performing Key Sensitivity Analysis...")
    
    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    image_row, image_col, channel = image_array.shape
    target_length = image_row * image_col * channel * 8

    # Generate first key (K1)
    original_key = "Bangladesh"
    key1_bits_original = ''.join(char_to_binary(c) for c in original_key)
    
    # Generate second key (K2) with a small change
    # In this case, we'll modify the last character slightly
    modified_key = "Bangladesg"  # 'h' changed to 'g'
    key1_bits_modified = ''.join(char_to_binary(c) for c in modified_key)
    
    # Generate all necessary keys for both original and modified
    
    
    # Expand keys to required length
    key1_bits_original = key_scramble(key1_bits_original, target_length)
    
    
    key1_bits_modified = key_scramble(key1_bits_modified, target_length)
    
    
    # Test I: Encryption Sensitivity Test
    print("\n--- I. Encryption Sensitivity Test ---")
    print("Encrypting the same image with two slightly different keys...")
    
    # Encrypt with K1
    encrypted_image1 = encryption(key1_bits_original, image_path)
    
    # Encrypt with K2
    encrypted_image2 = encryption(key1_bits_modified, image_path)
    
    # Calculate NPCR and UACI between the two encrypted images
    npcr_encryption = calculate_NPCR(encrypted_image1, encrypted_image2)
    uaci_encryption = calculate_UACI(encrypted_image1, encrypted_image2)
    
    print(f"NPCR between encrypted images: {npcr_encryption:.4f}%")
    print(f"UACI between encrypted images: {uaci_encryption:.4f}%")
    
    # Ideal values for reference
    print("Ideal values - NPCR: 99.6094%, UACI: 33.4635%")
    
    # Test II: Decryption Sensitivity
    print("\n--- II. Decryption Sensitivity Test ---")
    print("Encrypting with key K1 and attempting to decrypt with key K2...")
    
    # Decrypt with wrong key
    decrypted_with_wrong_key = decryption(key1_bits_modified, encrypted_image1)
    
    # Decrypt with correct key for comparison
    decrypted_with_correct_key = decryption(key1_bits_original, encrypted_image1)
    
    # Calculate difference between original image and incorrectly decrypted image
    npcr_decryption = calculate_NPCR(image_array, decrypted_with_wrong_key)
    uaci_decryption = calculate_UACI(image_array, decrypted_with_wrong_key)
    
    print(f"NPCR between original and incorrectly decrypted: {npcr_decryption:.4f}%")
    print(f"UACI between original and incorrectly decrypted: {uaci_decryption:.4f}%")
    
    # Check if correctly decrypted image matches original
    npcr_correct = calculate_NPCR(image_array, decrypted_with_correct_key)
    print(f"NPCR between original and correctly decrypted: {npcr_correct:.4f}%")
    
    # Visualize results - MODIFIED TO ONLY SHOW PLAIN IMAGE AND DECRYPTED IMAGES
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_array)
    plt.title("Original Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(decrypted_with_correct_key)
    plt.title("Decrypted with Correct Key")
    
    plt.subplot(1, 3, 3)
    plt.imshow(decrypted_with_wrong_key)
    plt.title("Decrypted with Wrong Key")
    
    plt.tight_layout()
    # plt.savefig("keysensetivity.png")
    plt.show()
    
    # Just show histograms for original image
    plt.figure(figsize=(8, 5))
    plt.hist(image_array.flatten(), bins=256, color='blue', alpha=0.7)
    plt.title("Original Image Histogram")
    plt.tight_layout()
    plt.show()
    
    # Summary of results
    print("\n--- Key Sensitivity Analysis Summary ---")
    print(f"1. Encryption Sensitivity: NPCR = {npcr_encryption:.4f}%, UACI = {uaci_encryption:.4f}%")
    print(f"2. Decryption Sensitivity: NPCR = {npcr_decryption:.4f}%, UACI = {uaci_decryption:.4f}%")
    
    if npcr_encryption > 99.5 and 33 < uaci_encryption < 34:
        print("✓ The algorithm shows excellent encryption sensitivity to key changes.")
    else:
        print("⚠ The encryption sensitivity may need improvement.")
        
    if npcr_decryption > 99.5:
        print("✓ The algorithm shows excellent decryption sensitivity to key changes.")
    else:
        print("⚠ The decryption sensitivity may need improvement.")
    
    return {
        'npcr_encryption': npcr_encryption,
        'uaci_encryption': uaci_encryption,
        'npcr_decryption': npcr_decryption,
        'uaci_decryption': uaci_decryption
    }

if __name__ == "__main__":
    # You can specify the path to your image here
    image_path = 'khalid/tree256.png'  # Replace with your actual image path
    
    # Run the key sensitivity analysis
    results = key_sensitivity_analysis(image_path)
