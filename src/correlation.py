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

# # ----- CORRELATION ANALYSIS FUNCTIONS -----

# def calculate_correlation_coefficient(x, y):
#     """
#     Calculate correlation coefficient using the formulas provided in the paper.
    
#     Parameters:
#     x, y: Lists of pixel values
    
#     Returns:
#     Correlation coefficient
#     """
#     # Convert to numpy arrays for vector operations
#     x = np.array(x)
#     y = np.array(y)
#     N = len(x)
    
#     # Equation 11: Mean/Expected value
#     E_x = np.sum(x) / N
#     E_y = np.sum(y) / N
    
#     # Equation 12: Standard deviation
#     S_x = np.sqrt(np.sum((x - E_x)**2) / N)
#     S_y = np.sqrt(np.sum((y - E_y)**2) / N)
    
#     # Equation 13: Covariance
#     cov_xy = np.sum((x - E_x) * (y - E_y)) / N
    
#     # Equation 14: Correlation coefficient
#     if S_x == 0 or S_y == 0:
#         return 0  # Handle edge case of zero standard deviation
    
#     r_xy = cov_xy / (S_x * S_y)
    
#     return r_xy

# def calculate_pixel_correlation(image_array, sample_size=2000):
#     """
#     Calculate the correlation between adjacent pixels in an image.
    
#     Parameters:
#     image_array: numpy array of the image
#     sample_size: number of pixel pairs to sample
    
#     Returns:
#     Dictionary containing correlation coefficients and pixel pairs for plotting
#     """
#     height, width, channels = image_array.shape
#     results = {}
    
#     for channel_idx, channel_name in enumerate(['Red', 'Green', 'Blue']):
#         # Extract the specific color channel
#         channel = image_array[:, :, channel_idx]
        
#         # Prepare containers for pixel pairs
#         horizontal_pairs = {'x': [], 'y': []}
#         vertical_pairs = {'x': [], 'y': []}
#         diagonal_pairs = {'x': [], 'y': []}
        
#         # Randomly sample pixel pairs
#         for _ in range(sample_size):
#             # Select random pixel position (avoiding edges)
#             i = random.randint(0, height - 2)
#             j = random.randint(0, width - 2)
            
#             # Current pixel value
#             pixel = channel[i, j]
            
#             # Adjacent pixel values
#             horizontal_neighbor = channel[i, j + 1]
#             vertical_neighbor = channel[i + 1, j]
#             diagonal_neighbor = channel[i + 1, j + 1]
            
#             # Store pixel pairs
#             horizontal_pairs['x'].append(pixel)
#             horizontal_pairs['y'].append(horizontal_neighbor)
            
#             vertical_pairs['x'].append(pixel)
#             vertical_pairs['y'].append(vertical_neighbor)
            
#             diagonal_pairs['x'].append(pixel)
#             diagonal_pairs['y'].append(diagonal_neighbor)
        
#         # Calculate correlation coefficients using the equations from the paper
#         results[channel_name] = {
#             'horizontal': {
#                 'pairs': horizontal_pairs,
#                 'correlation': calculate_correlation_coefficient(horizontal_pairs['x'], horizontal_pairs['y'])
#             },
#             'vertical': {
#                 'pairs': vertical_pairs,
#                 'correlation': calculate_correlation_coefficient(vertical_pairs['x'], vertical_pairs['y'])
#             },
#             'diagonal': {
#                 'pairs': diagonal_pairs,
#                 'correlation': calculate_correlation_coefficient(diagonal_pairs['x'], diagonal_pairs['y'])
#             }
#         }
    
#     return results

# def plot_correlation_comparison(original_correlation, encrypted_correlation):
#     """
#     Plot scatter plots of pixel correlations for both plain and cipher images in a single figure,
#     similar to the Fig. 6 in the paper.
    
#     Parameters:
#     original_correlation: Correlation results from the original image
#     encrypted_correlation: Correlation results from the encrypted image
#     """
#     # Create a 3x6 grid of subplots (3 channels x 3 directions x 2 types)
#     fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    
#     # Set up channel colors for plots
#     channel_colors = {'Red': 'red', 'Green': 'green', 'Blue': 'blue'}
    
#     # Column titles and labels
#     directions = ['Horizontal', 'Vertical', 'Diagonal']
    
#     # Set column titles
#     for i, direction in enumerate(directions):
#         col_idx = i * 2  # 0, 2, 4
#         axes[0, col_idx].set_title(f"{direction} Correlation\nPlain")
#         axes[0, col_idx + 1].set_title(f"{direction} Correlation\nCipher")
    
#     # Add subplot labels similar to the paper (a, b, c, ...)
#     labels = [
#         ['(a)', '(d)', '(g)', '(j)', '(m)', '(p)'],
#         ['(b)', '(e)', '(h)', '(k)', '(n)', '(q)'],
#         ['(c)', '(f)', '(i)', '(l)', '(o)', '(r)']
#     ]
    
#     # Iterate through each channel and direction
#     for row, (channel, color) in enumerate(channel_colors.items()):
#         for dir_idx, direction in enumerate([d.lower() for d in directions]):
#             # Calculate column indices for plain and cipher
#             plain_col = dir_idx * 2
#             cipher_col = dir_idx * 2 + 1
            
#             # Extract data for plain image
#             plain_data = original_correlation[channel][direction]
#             plain_x = plain_data['pairs']['x']
#             plain_y = plain_data['pairs']['y']
#             plain_corr = plain_data['correlation']
            
#             # Extract data for cipher image
#             cipher_data = encrypted_correlation[channel][direction]
#             cipher_x = cipher_data['pairs']['x']
#             cipher_y = cipher_data['pairs']['y']
#             cipher_corr = cipher_data['correlation']
            
#             # Plot plain image correlation
#             axes[row, plain_col].scatter(plain_x, plain_y, s=5, alpha=0.7, c=color, edgecolors='none')
#             axes[row, plain_col].set_xlim([0, 255])
#             axes[row, plain_col].set_ylim([0, 255])
#             axes[row, plain_col].text(10, 240, f"r = {plain_corr:.6f}", fontsize=8, 
#                                      bbox=dict(facecolor='white', alpha=0.7))
#             axes[row, plain_col].set_xlabel(labels[row][plain_col])
            
#             # Plot cipher image correlation
#             axes[row, cipher_col].scatter(cipher_x, cipher_y, s=5, alpha=0.7, c=color, edgecolors='none')
#             axes[row, cipher_col].set_xlim([0, 255])
#             axes[row, cipher_col].set_ylim([0, 255])
#             axes[row, cipher_col].text(10, 240, f"r = {cipher_corr:.6f}", fontsize=8, 
#                                       bbox=dict(facecolor='white', alpha=0.7))
#             axes[row, cipher_col].set_xlabel(labels[row][cipher_col])
            
#             # Label only the leftmost plots with channel name
#             if dir_idx == 0:
#                 axes[row, plain_col].set_ylabel(channel)
    
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     # plt.suptitle("Correlation analysis of Lena plain and cipher images", fontsize=16)
    
#     return fig

# def analyze_image_correlation(original_image, encrypted_image=None):
#     """
#     Analyze and compare the correlation of original and encrypted images.
    
#     Parameters:
#     original_image: Original image array
#     encrypted_image: Encrypted image array (if provided)
#     """
#     # Calculate correlation for original image
#     original_correlation = calculate_pixel_correlation(original_image)
    
#     if encrypted_image is not None:
#         # Calculate correlation for encrypted image
#         encrypted_correlation = calculate_pixel_correlation(encrypted_image)
        
#         # Plot correlation comparison (both plain and cipher in one figure)
#         fig = plot_correlation_comparison(original_correlation, encrypted_correlation)
#         plt.savefig("correlation_analysis.png", dpi=300, bbox_inches='tight')
#         plt.show()
        
#         # Print summary of correlation coefficients
#         print("Correlation Coefficients Summary:")
#         print("-" * 60)
#         print(f"{'Direction':<12}{'Channel':<8}{'Plain':<15}{'Cipher':<15}")
#         print("-" * 60)
        
#         for direction in ['horizontal', 'vertical', 'diagonal']:
#             for channel in ['Red', 'Green', 'Blue']:
#                 plain_corr = original_correlation[channel][direction]['correlation']
#                 cipher_corr = encrypted_correlation[channel][direction]['correlation']
#                 print(f"{direction.capitalize():<12}{channel:<8}{plain_corr:<15.6f}{cipher_corr:<15.6f}")
        
#         # Print H, V, D summary (average across channels)
#         print("\nAverage Correlation Coefficients (H, V, D):")
#         print("-" * 42)
#         h_plain = sum(original_correlation[c]['horizontal']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
#         v_plain = sum(original_correlation[c]['vertical']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
#         d_plain = sum(original_correlation[c]['diagonal']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
        
#         h_cipher = sum(encrypted_correlation[c]['horizontal']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
#         v_cipher = sum(encrypted_correlation[c]['vertical']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
#         d_cipher = sum(encrypted_correlation[c]['diagonal']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
        
#         print(f"H: {h_plain:.6f} (plain) → {h_cipher:.6f} (cipher)")
#         print(f"V: {v_plain:.6f} (plain) → {v_cipher:.6f} (cipher)")
#         print(f"D: {d_plain:.6f} (plain) → {d_cipher:.6f} (cipher)")
#     else:
#         # If no encrypted image is provided, only plot original correlation
#         # Use plain scatter plot function
#         fig, ax = plt.subplots(figsize=(10, 8))
#         for channel_name in ['Red', 'Green', 'Blue']:
#             data = original_correlation[channel_name]['horizontal']
#             ax.scatter(data['pairs']['x'], data['pairs']['y'], s=5, alpha=0.7, 
#                       label=f"{channel_name} (r={data['correlation']:.6f})")
#         ax.set_xlim([0, 255])
#         ax.set_ylim([0, 255])
#         ax.set_title("Plain Image Horizontal Correlation")
#         ax.legend()
#         plt.savefig("plain_correlation.png")
#         plt.show()

# def main():
#     """Main function to run the encryption and analysis."""
#     image_path = 'khalid/Lena256.png'  # Replace with your image path
#     image = Image.open(image_path)
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
#     cipherImage = encryption(key1_bits, key2_bits, key3_bits, image_path)
    
#     # # Display the encrypted image
#     # plt.figure(figsize=(10, 8))
#     # plt.imshow(cipherImage)
#     # plt.title("Encrypted Image")
#     # plt.axis('off')
#     # # plt.savefig("encrypted_image.png")
#     # plt.show()
    
#     # Calculate and display entropy
#     original_1d = image_array.flatten()
#     print("Original Image Entropy:", calculate_entropy(original_1d)) 
    
#     cipherImage_1d = cipherImage.flatten()
#     print("Encrypted Image Entropy:", calculate_entropy(cipherImage_1d))
    
#     # Plot histograms
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.hist(original_1d, bins=256, color='blue', alpha=0.7)
#     plt.title("Original Image Histogram")
#     plt.xlabel("Pixel Value")
#     plt.ylabel("Frequency")
    
#     plt.subplot(1, 2, 2)
#     plt.hist(cipherImage_1d, bins=256, color='red', alpha=0.7)
#     plt.title("Encrypted Image Histogram")
#     plt.xlabel("Pixel Value")
#     plt.ylabel("Frequency")
    
#     plt.tight_layout()
#     # plt.savefig("histograms.png")
#     plt.show()
    
#     # Run correlation analysis
#     analyze_image_correlation(image_array, cipherImage)
    
#     # # Decrypt the image to verify
#     # decrypted_image = decryption(key1_bits, key2_bits, key3_bits, cipherImage)
    
#     # # Display the decrypted image
#     # plt.figure(figsize=(10, 8))
#     # plt.imshow(decrypted_image)
#     # plt.title("Decrypted Image")
#     # plt.axis('off')
#     # # plt.savefig("decrypted_image.png")
#     # plt.show()
    
    
# if __name__ == "__main__":
#     main()







import math
import numpy as np 
from collections import Counter
from PIL import Image
import random
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
# ----- CORRELATION ANALYSIS FUNCTIONS -----

def calculate_correlation_coefficient(x, y):
    """
    Calculate correlation coefficient using the formulas provided in the paper.
    
    Parameters:
    x, y: Lists of pixel values
    
    Returns:
    Correlation coefficient
    """
    # Convert to numpy arrays for vector operations
    x = np.array(x)
    y = np.array(y)
    N = len(x)
    
    # Equation 11: Mean/Expected value
    E_x = np.sum(x) / N
    E_y = np.sum(y) / N
    
    # Equation 12: Standard deviation
    S_x = np.sqrt(np.sum((x - E_x)**2) / N)
    S_y = np.sqrt(np.sum((y - E_y)**2) / N)
    
    # Equation 13: Covariance
    cov_xy = np.sum((x - E_x) * (y - E_y)) / N
    
    # Equation 14: Correlation coefficient
    if S_x == 0 or S_y == 0:
        return 0  # Handle edge case of zero standard deviation
    
    r_xy = cov_xy / (S_x * S_y)
    
    return r_xy

def calculate_pixel_correlation(image_array, sample_size=2000):
    """
    Calculate the correlation between adjacent pixels in an image.
    
    Parameters:
    image_array: numpy array of the image
    sample_size: number of pixel pairs to sample
    
    Returns:
    Dictionary containing correlation coefficients and pixel pairs for plotting
    """
    height, width, channels = image_array.shape
    results = {}
    
    for channel_idx, channel_name in enumerate(['Red', 'Green', 'Blue']):
        # Extract the specific color channel
        channel = image_array[:, :, channel_idx]
        
        # Prepare containers for pixel pairs
        horizontal_pairs = {'x': [], 'y': []}
        vertical_pairs = {'x': [], 'y': []}
        diagonal_pairs = {'x': [], 'y': []}
        
        # Randomly sample pixel pairs
        for _ in range(sample_size):
            # Select random pixel position (avoiding edges)
            i = random.randint(0, height - 2)
            j = random.randint(0, width - 2)
            
            # Current pixel value
            pixel = channel[i, j]
            
            # Adjacent pixel values
            horizontal_neighbor = channel[i, j + 1]
            vertical_neighbor = channel[i + 1, j]
            diagonal_neighbor = channel[i + 1, j + 1]
            
            # Store pixel pairs
            horizontal_pairs['x'].append(pixel)
            horizontal_pairs['y'].append(horizontal_neighbor)
            
            vertical_pairs['x'].append(pixel)
            vertical_pairs['y'].append(vertical_neighbor)
            
            diagonal_pairs['x'].append(pixel)
            diagonal_pairs['y'].append(diagonal_neighbor)
        
        # Calculate correlation coefficients using the equations from the paper
        results[channel_name] = {
            'horizontal': {
                'pairs': horizontal_pairs,
                'correlation': calculate_correlation_coefficient(horizontal_pairs['x'], horizontal_pairs['y'])
            },
            'vertical': {
                'pairs': vertical_pairs,
                'correlation': calculate_correlation_coefficient(vertical_pairs['x'], vertical_pairs['y'])
            },
            'diagonal': {
                'pairs': diagonal_pairs,
                'correlation': calculate_correlation_coefficient(diagonal_pairs['x'], diagonal_pairs['y'])
            }
        }
    
    return results

def plot_correlation_comparison(original_correlation, encrypted_correlation):
    """
    Plot scatter plots of pixel correlations for both plain and cipher images in a single figure,
    similar to the Fig. 6 in the paper.
    
    Parameters:
    original_correlation: Correlation results from the original image
    encrypted_correlation: Correlation results from the encrypted image
    """
    # Create a 3x6 grid of subplots (3 channels x 3 directions x 2 types)
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    
    # Set up channel colors for plots
    channel_colors = {'Red': 'red', 'Green': 'green', 'Blue': 'blue'}
    
    # Column titles and labels
    directions = ['Horizontal', 'Vertical', 'Diagonal']
    
    # Set column titles
    for i, direction in enumerate(directions):
        col_idx = i * 2  # 0, 2, 4
        axes[0, col_idx].set_title(f"{direction} Correlation\nPlain")
        axes[0, col_idx + 1].set_title(f"{direction} Correlation\nCipher")
    
    # Add subplot labels similar to the paper (a, b, c, ...)
    labels = [
        ['(a)', '(d)', '(g)', '(j)', '(m)', '(p)'],
        ['(b)', '(e)', '(h)', '(k)', '(n)', '(q)'],
        ['(c)', '(f)', '(i)', '(l)', '(o)', '(r)']
    ]
    
    # Iterate through each channel and direction
    for row, (channel, color) in enumerate(channel_colors.items()):
        for dir_idx, direction in enumerate([d.lower() for d in directions]):
            # Calculate column indices for plain and cipher
            plain_col = dir_idx * 2
            cipher_col = dir_idx * 2 + 1
            
            # Extract data for plain image
            plain_data = original_correlation[channel][direction]
            plain_x = plain_data['pairs']['x']
            plain_y = plain_data['pairs']['y']
            plain_corr = plain_data['correlation']
            
            # Extract data for cipher image
            cipher_data = encrypted_correlation[channel][direction]
            cipher_x = cipher_data['pairs']['x']
            cipher_y = cipher_data['pairs']['y']
            cipher_corr = cipher_data['correlation']
            
            # Plot plain image correlation
            axes[row, plain_col].scatter(plain_x, plain_y, s=5, alpha=0.7, c=color, edgecolors='none')
            axes[row, plain_col].set_xlim([0, 255])
            axes[row, plain_col].set_ylim([0, 255])
            axes[row, plain_col].text(10, 240, f"r = {plain_corr:.6f}", fontsize=8, 
                                     bbox=dict(facecolor='white', alpha=0.7))
            axes[row, plain_col].set_xlabel(labels[row][plain_col])
            
            # Plot cipher image correlation
            axes[row, cipher_col].scatter(cipher_x, cipher_y, s=5, alpha=0.7, c=color, edgecolors='none')
            axes[row, cipher_col].set_xlim([0, 255])
            axes[row, cipher_col].set_ylim([0, 255])
            axes[row, cipher_col].text(10, 240, f"r = {cipher_corr:.6f}", fontsize=8, 
                                      bbox=dict(facecolor='white', alpha=0.7))
            axes[row, cipher_col].set_xlabel(labels[row][cipher_col])
            
            # Label only the leftmost plots with channel name
            if dir_idx == 0:
                axes[row, plain_col].set_ylabel(channel)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.suptitle("Correlation analysis of Lena plain and cipher images", fontsize=16)
    
    return fig

def analyze_image_correlation(original_image, encrypted_image=None):
    """
    Analyze and compare the correlation of original and encrypted images.
    
    Parameters:
    original_image: Original image array
    encrypted_image: Encrypted image array (if provided)
    """
    # Calculate correlation for original image
    original_correlation = calculate_pixel_correlation(original_image)
    
    if encrypted_image is not None:
        # Calculate correlation for encrypted image
        encrypted_correlation = calculate_pixel_correlation(encrypted_image)
        
        # Plot correlation comparison (both plain and cipher in one figure)
        fig = plot_correlation_comparison(original_correlation, encrypted_correlation)
        # plt.savefig("correlation_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary of correlation coefficients
        print("Correlation Coefficients Summary:")
        print("-" * 60)
        print(f"{'Direction':<12}{'Channel':<8}{'Plain':<15}{'Cipher':<15}")
        print("-" * 60)
        
        for direction in ['horizontal', 'vertical', 'diagonal']:
            for channel in ['Red', 'Green', 'Blue']:
                plain_corr = original_correlation[channel][direction]['correlation']
                cipher_corr = encrypted_correlation[channel][direction]['correlation']
                print(f"{direction.capitalize():<12}{channel:<8}{plain_corr:<15.6f}{cipher_corr:<15.6f}")
        
        # Print H, V, D summary (average across channels)
        print("\nAverage Correlation Coefficients (H, V, D):")
        print("-" * 42)
        h_plain = sum(original_correlation[c]['horizontal']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
        v_plain = sum(original_correlation[c]['vertical']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
        d_plain = sum(original_correlation[c]['diagonal']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
        
        h_cipher = sum(encrypted_correlation[c]['horizontal']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
        v_cipher = sum(encrypted_correlation[c]['vertical']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
        d_cipher = sum(encrypted_correlation[c]['diagonal']['correlation'] for c in ['Red', 'Green', 'Blue']) / 3
        
        print(f"H: {h_plain:.6f} (plain) → {h_cipher:.6f} (cipher)")
        print(f"V: {v_plain:.6f} (plain) → {v_cipher:.6f} (cipher)")
        print(f"D: {d_plain:.6f} (plain) → {d_cipher:.6f} (cipher)")
    else:
        # If no encrypted image is provided, only plot original correlation
        # Use plain scatter plot function
        fig, ax = plt.subplots(figsize=(10, 8))
        for channel_name in ['Red', 'Green', 'Blue']:
            data = original_correlation[channel_name]['horizontal']
            ax.scatter(data['pairs']['x'], data['pairs']['y'], s=5, alpha=0.7, 
                      label=f"{channel_name} (r={data['correlation']:.6f})")
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_title("Plain Image Horizontal Correlation")
        ax.legend()
        plt.savefig("plain_correlation.png")
        plt.show()

def main():
    """Main function to run the encryption and analysis."""
    image_path = 'khalid/female256.png'  # Replace with your image path
    image = Image.open(image_path)
    image_array = np.array(image)
    
    image_row, image_col, channel = image_array.shape
    target_length = image_row * image_col * channel * 8

    key = "Bangladesh"
    key1_bits = ''.join(char_to_binary(c) for c in key)

    

    key1_bits = key_scramble(key1_bits, target_length)
    
    
    # Encrypt the image
    cipherImage = encryption(key1_bits, image_path)
    
    # # Display the encrypted image
    # plt.figure(figsize=(10, 8))
    # plt.imshow(cipherImage)
    # plt.title("Encrypted Image")
    # plt.axis('off')
    # # plt.savefig("encrypted_image.png")
    # plt.show()
    
    # Calculate and display entropy
    original_1d = image_array.flatten()
    print("Original Image Entropy:", calculate_entropy(original_1d)) 
    
    cipherImage_1d = cipherImage.flatten()
    print("Encrypted Image Entropy:", calculate_entropy(cipherImage_1d))
    
    # Plot histograms
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(original_1d, bins=256, color='blue', alpha=0.7)
    plt.title("Original Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(cipherImage_1d, bins=256, color='red', alpha=0.7)
    plt.title("Encrypted Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    # plt.savefig("histograms.png")
    plt.show()
    
    # Run correlation analysis
    analyze_image_correlation(image_array, cipherImage)
    
    # # Decrypt the image to verify
    # decrypted_image = decryption(key1_bits, key2_bits, key3_bits, cipherImage)
    
    # # Display the decrypted image
    # plt.figure(figsize=(10, 8))
    # plt.imshow(decrypted_image)
    # plt.title("Decrypted Image")
    # plt.axis('off')
    # # plt.savefig("decrypted_image.png")
    # plt.show()
    
    
if __name__ == "__main__":
    main()



