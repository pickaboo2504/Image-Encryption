import math
import numpy as np
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import random
import hashlib
from skimage.metrics import structural_similarity as ssim

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

def crop_image(image, crop_percentage):
    """Crop an image by the given percentage from the center."""
    height, width, channels = image.shape
    crop_height = int(height * crop_percentage / 100)
    crop_width = int(width * crop_percentage / 100)
    
    start_h = height // 2 - crop_height // 2
    end_h = start_h + crop_height 
    start_w = width // 2 - crop_width // 2
    end_w = start_w + crop_width
    
    cropped_image = image.copy()
    cropped_image[start_h:end_h, start_w:end_w] = 0  # Fill with black
    
    return cropped_image

def calculate_psnr(original_image, processed_image):
    """Calculate PSNR using the formula Ps = 10*log10(Mx/Me)."""
    if original_image.shape != processed_image.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Mean square error (Me)
    mse = np.mean((original_image.astype(float) - processed_image.astype(float)) ** 2)
    if mse == 0:  # Images are identical
        return float('inf')
    
    # Maximum pixel value (Mx)
    max_pixel = np.max(original_image)
    
    # PSNR calculation
    psnr = 10 * np.log10(max_pixel**2 / mse)
    return psnr

def calculate_ssim(original_image, processed_image):
    """Calculate SSIM using skimage's implementation."""
    if original_image.shape != processed_image.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to grayscale if needed
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        # Calculate SSIM for each channel and take average
        ssim_value = 0
        for i in range(3):
            ssim_value += ssim(original_image[:,:,i], processed_image[:,:,i], 
                             data_range=original_image.max() - original_image.min())
        ssim_value /= 3
    else:
        ssim_value = ssim(original_image, processed_image, 
                        data_range=original_image.max() - original_image.min())
    
    return ssim_value

def main():
    # Load image (using a sample image path - replace with your own)
    image_path = 'khalid/baboon.png'  # Change this to your image path
    
    # For testing without a real image, uncomment these lines:
    # Generate a simple test image
    image_array = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            image_array[i, j, 0] = i % 256  # R
            image_array[i, j, 1] = j % 256  # G
            image_array[i, j, 2] = (i+j) % 256  # B
    
    # Create PIL image from array
    image = Image.fromarray(image_array)
    
    # Uncomment to use a real image instead
    # image = Image.open(image_path)
    # image_array = np.array(image)
    
    # Calculate key lengths
    image_row, image_col, channel = image_array.shape
    target_length = image_row * image_col * channel * 8

    # Generate keys
    key = "Bangladesh"
    key1_bits = ''.join(char_to_binary(c) for c in key)
    key2_bits, key3_bits = generate_keys(key1_bits)
    key1_bits = key_scramble(key1_bits, target_length)
    key2_bits = key_scramble(key2_bits, target_length)
    key3_bits = key_scramble(key3_bits, target_length)
    
    # Encrypt the image
    cipher_image = encryption(key1_bits, key2_bits, key3_bits, image_array)
    
    # Create cropped versions at different percentages
    crop_percentages = [10, 20, 25, 50]
    cropped_images = []
    
    for percentage in crop_percentages:
        cropped = crop_image(cipher_image, percentage)
        cropped_images.append((percentage, cropped))
    
    # Decrypt the cropped images
    decrypted_images = []
    
    for percentage, cropped_img in cropped_images:
        decrypted = decryption(key1_bits, key2_bits, key3_bits, cropped_img)
        decrypted_images.append((percentage, decrypted))
    
    # Calculate metrics
    metrics_data = []
    
    for percentage, decrypted_img in decrypted_images:
        psnr_value = calculate_psnr(image_array, decrypted_img)
        ssim_value = calculate_ssim(image_array, decrypted_img)
        metrics_data.append((percentage, psnr_value, ssim_value))
    
    # Display original, encrypted, and all cropped versions with their decrypted counterparts
    # For 4 cropping percentages we need 10 subplots in total (original + encrypted + 4 cropped + 4 decrypted)
    plt.figure(figsize=(15, 12))
    
    # Original and encrypted images (row 1)
    plt.subplot(3, 4, 1)
    plt.imshow(image_array)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(cipher_image)
    plt.title("Encrypted Image")
    plt.axis('off')
    
    # Display cropped images (row 2)
    for i, (percentage, cropped_img) in enumerate(cropped_images):
        plt.subplot(3, 4, 5 + i)
        plt.imshow(cropped_img)
        plt.title(f"Cropped {percentage}%")
        plt.axis('off')
    
    # Display decrypted images (row 3)
    for i, (percentage, decrypted_img) in enumerate(decrypted_images):
        plt.subplot(3, 4, 9 + i)
        plt.imshow(decrypted_img)
        plt.title(f"Decrypted after {percentage}% crop")
        plt.axis('off')
    
    plt.tight_layout()
    #plt.savefig("cropping_analysis.png")
    plt.show()
    
    # Print metrics data in a table
    print("\nMetrics for Different Cropping Percentages:")
    print("-" * 60)
    print(f"{'Cropping %':<15} {'PSNR':<15} {'SSIM':<15}")
    print("-" * 60)
    
    for percentage, psnr_value, ssim_value in metrics_data:
        print(f"{percentage:<15} {psnr_value:<15.4f} {ssim_value:<15.4f}")
    
    # Create plots for metrics
    plt.figure(figsize=(12, 5))
    
    # PSNR plot
    plt.subplot(1, 2, 1)
    percentages = [data[0] for data in metrics_data]
    psnr_values = [data[1] for data in metrics_data]
    plt.plot(percentages, psnr_values, 'o-', color='blue')
    plt.title("PSNR vs Cropping Percentage")
    plt.xlabel("Cropping Percentage")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    
    # SSIM plot
    plt.subplot(1, 2, 2)
    ssim_values = [data[2] for data in metrics_data]
    plt.plot(percentages, ssim_values, 'o-', color='green')
    plt.title("SSIM vs Cropping Percentage")
    plt.xlabel("Cropping Percentage")
    plt.ylabel("SSIM")
    plt.grid(True)
    
    plt.tight_layout()
    #plt.savefig("metrics_plot.png")
    plt.show()

if __name__ == "__main__":
    main()