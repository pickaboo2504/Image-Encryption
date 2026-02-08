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

def shift_pixels(image_array, shift_value):
    """Shift image pixels by a specific value."""
    shifted_image = np.roll(image_array, shift_value, axis=(0, 1))
    return shifted_image

def generate_shift_value(key):
    """Generate an unpredictable shift value between 1-100 based on key."""
    hash_value = hashlib.sha256(key.encode()).hexdigest()
    # Convert part of hash to integer in range 1-100
    shift_value = int(hash_value[:8], 16) % 100 + 1  # Ensures value is between 1-100
    return shift_value

def encryption(key, image_path):
    """Encrypt an image using single-layer DNA encoding."""
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Handle grayscale images
        if len(image_array.shape) == 2:
            # Convert grayscale to 3D array with single channel
            image_array = np.expand_dims(image_array, axis=2)
            
        height, width = image_array.shape[0], image_array.shape[1]
        
        # Determine number of channels (handle both RGB and grayscale)
        if len(image_array.shape) > 2:
            channels = image_array.shape[2]
        else:
            channels = 1
            
        # Apply pixel shifting
        shift_value = generate_shift_value(key)
        image_array = shift_pixels(image_array, shift_value)
        
        processed_image = np.zeros(shape=(height, width, channels), dtype=np.uint8)

        index = 0
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    if index < len(key):
                        rule = (index % 8) + 1
                        key_segment = key[index*8:(index+1)*8]
                        
                        original_bin = format(image_array[i, j, k], '08b')
                        
                        # Convert to DNA, apply XOR, and convert back
                        dna_original = DNAEncode(original_bin, rule)
                        dna_key = DNAEncode(key_segment, rule)
                        dna_encrypted = DNAXOR(dna_original, dna_key)
                        encrypted_bin = DNADecode(dna_encrypted, rule)
                        
                        processed_image[i, j, k] = int(encrypted_bin, 2)
                        
                        index += 1
        return processed_image
    except Exception as e:
        print(f"Error during encryption: {e}")
        import traceback
        traceback.print_exc()
        raise

def decryption(key, encrypted_image):
    """Decrypt an image using single-layer DNA decoding."""
    try:
        cipher_array = np.array(encrypted_image)
        
        # Handle dimensionality
        if len(cipher_array.shape) == 2:
            # Convert grayscale to 3D array with single channel
            cipher_array = np.expand_dims(cipher_array, axis=2)
            
        height, width = cipher_array.shape[0], cipher_array.shape[1]
        
        # Determine number of channels (handle both RGB and grayscale)
        if len(cipher_array.shape) > 2:
            channels = cipher_array.shape[2]
        else:
            channels = 1
            
        decrypted_image = np.zeros(shape=(height, width, channels), dtype=np.uint8)

        index = 0
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    if index < len(key):
                        rule = (index % 8) + 1
                        key_segment = key[index*8:(index+1)*8]
                        
                        encrypted_bin = format(cipher_array[i, j, k], '08b')
                        
                        # Convert to DNA, apply XOR again (reverses the encryption), and convert back
                        dna_encrypted = DNAEncode(encrypted_bin, rule)
                        dna_key = DNAEncode(key_segment, rule)
                        dna_original = DNAXOR(dna_encrypted, dna_key)
                        original_bin = DNADecode(dna_original, rule)
                        
                        decrypted_image[i, j, k] = int(original_bin, 2)
                        
                        index += 1
        
        # Reverse the pixel shifting
        shift_value = generate_shift_value(key)
        decrypted_image = shift_pixels(decrypted_image, -shift_value)
        
        return decrypted_image
    except Exception as e:
        print(f"Error during decryption: {e}")
        import traceback
        traceback.print_exc()
        raise

def calculate_ssim(imageA, imageB):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    """
    # Convert images to float for calculations
    imageA = imageA.astype(np.float64)
    imageB = imageB.astype(np.float64)
    
    # Constants to avoid instability when denominator is close to zero
    L = 255.0  # For 8-bit images
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    # Calculate means
    mu_A = np.mean(imageA)
    mu_B = np.mean(imageB)
    
    # Calculate standard deviations
    sigma_A = np.std(imageA)
    sigma_B = np.std(imageB)
    
    # Calculate covariance
    sigma_AB = np.mean((imageA - mu_A) * (imageB - mu_B))
    
    # Calculate SSIM
    numerator = (2 * mu_A * mu_B + c1) * (2 * sigma_AB + c2)
    denominator = (mu_A**2 + mu_B**2 + c1) * (sigma_A**2 + sigma_B**2 + c2)
    ssim = numerator / denominator
    
    return ssim

def calculate_ssim_per_channel(imageA, imageB):
    """Calculate SSIM for each channel of RGB images."""
    # Make sure images are numpy arrays
    imageA = np.array(imageA)
    imageB = np.array(imageB)
    
    # If images are grayscale, return single SSIM
    if len(imageA.shape) == 2 or imageA.shape[2] == 1:
        return {"Gray": calculate_ssim(imageA, imageB)}
    
    # For RGB images, calculate SSIM for each channel
    ssim_values = {}
    channel_names = ['R', 'G', 'B']  # Default channel names for RGB
    
    # Make sure we only process available channels
    num_channels = min(imageA.shape[2], len(channel_names))
    
    for i in range(num_channels):
        channel_ssim = calculate_ssim(imageA[:,:,i], imageB[:,:,i])
        ssim_values[channel_names[i]] = channel_ssim
    
    # If there are more channels than names, use numeric indices
    for i in range(len(channel_names), imageA.shape[2]):
        channel_ssim = calculate_ssim(imageA[:,:,i], imageB[:,:,i])
        ssim_values[f"Channel_{i}"] = channel_ssim
    
    return ssim_values

def main():
    try:
        image_path = 'khalid/Baboon512.png'  # Update with your image path
        try:
            image = Image.open(image_path)
            print(f"Successfully opened image: {image_path}")
        except Exception as e:
            print(f"Error opening image file: {e}")
            return
            
        image_array = np.array(image)
        print(f"Image shape: {image_array.shape}")
        
        if len(image_array.shape) < 3:
            print("Image is grayscale. Adding channel dimension.")
            image_array = np.expand_dims(image_array, axis=2)
            channels = 1
        else:
            channels = image_array.shape[2]
        
        image_row, image_col = image_array.shape[0], image_array.shape[1]
        print(f"Image dimensions: {image_row}x{image_col}x{channels}")
        
        target_length = image_row * image_col * channels * 8
        print(f"Target key length: {target_length}")

        key = "Bangladesh"
        key_bits = ''.join(char_to_binary(c) for c in key)
        print(f"Original key length: {len(key_bits)} bits")

        scrambled_key_bits = key_scramble(key_bits, target_length)
        print(f"Scrambled key length: {len(scrambled_key_bits)} bits")
        
        print("Encrypting image...")
        cipherImage = encryption(scrambled_key_bits, image_path)
        print("Encryption complete!")
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image_array)
        plt.title("Original Image")
        
        plt.subplot(1, 3, 2)
        plt.imshow(cipherImage)
        plt.title("Encrypted Image")
        
        # Calculate and display entropy
        original_1d = image_array.flatten()
        original_entropy = calculate_entropy(original_1d)
        print("Original Image Entropy:", original_entropy) 
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(original_1d, bins=256)
        plt.title("Original Image Histogram")

        cipherImage_1d = cipherImage.flatten()
        cipher_entropy = calculate_entropy(cipherImage_1d)
        print("Encrypted Image Entropy:", cipher_entropy)
        plt.subplot(1, 2, 2)
        plt.hist(cipherImage_1d, bins=256)
        plt.title("Encrypted Image Histogram")
        
        # Calculate SSIM between original and encrypted images
        print("Calculating SSIM...")
        ssim_values = calculate_ssim_per_channel(image_array, cipherImage)
        print("SSIM values between original and encrypted images:")
        for channel, value in ssim_values.items():
            print(f"{channel}: {value:.4f}")
        
        # Decrypt image
        print("Decrypting image...")
        decrypted_image = decryption(scrambled_key_bits, cipherImage)
        print("Decryption complete!")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(decrypted_image)
        plt.title("Decrypted Image")
        
        # Verify decryption quality by comparing with original
        ssim_decrypted = calculate_ssim_per_channel(image_array, decrypted_image)
        print("\nSSIM values between original and decrypted images:")
        for channel, value in ssim_decrypted.items():
            print(f"{channel}: {value:.4f}")
        
        plt.show()
    except Exception as e:
        import traceback
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()