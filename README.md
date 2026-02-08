

```markdown
# A novel pixel-shifting and key-scrambling driven triple-layer key generation technique for DNA-based image encryption

This repository contains the Python implementation of the image encryption framework presented in our IPOL paper: *[A novel pixel-shifting and key-scrambling driven triple-layer key generation technique for DNA-based image encryption]*. The main code is in `MultiLayer_BaseCode.py`, and additional scripts are included for testing robustness, randomness, and image quality metrics.

---

## Repository Structure

```

.
├── data/                  # Input images for testing
├── src/                   # Python source code
│   ├── MultiLayer_BaseCode.py   # Main encryption code
│   ├── chi_square.py           # Chi-square test for encrypted images
│   ├── compression.py          # Test robustness under compression
│   ├── reconstruction.py       # Test image reconstruction after decryption
│   ├── keyVariableLength.py    # Test encryption with keys of different lengths
│   ├── PSNR.py                 # Peak Signal-to-Noise Ratio evaluation
│   ├── dataLoss.py             # Test robustness against data loss
│   ├── runTestRandomness.py    # Run multiple randomness tests (entropy, correlation, chi-square)
│   ├── SSMI.py                 # Structural similarity metric evaluation
│   ├── correlation.py          # Compute horizontal, vertical, and diagonal correlations
│   ├── Gaussian.py             # Gaussian noise testing
│   ├── salt&pepper.py          # Salt-and-pepper noise testing
│   ├── keySensitivity.py       # Key sensitivity tests
│   ├── NPCRandUACI.py          # NPCR and UACI metrics
│   ├── dataConvert.py          # Utilities for data conversion and preprocessing
│   └── Dynamic.py              # Dynamic testing utilities
├── requirements.txt       # Python dependencies
└── README.md              # This file

````

---

## Environment Setup

The code is written in **Python 3.8+**. Install all dependencies using:

```bash
pip install -r requirements.txt
````

Dependencies include:

* `numpy`
* `opencv-python`
* `scipy`
* `matplotlib`
* `Pillow`
* `tqdm`
* `hashlib`

*It is recommended to use a virtual environment to avoid conflicts.*

---

## How to Use

### 1. Encrypt an Image

The main encryption routine is in `MultiLayer_BaseCode.py`:

```bash
python src/MultiLayer_BaseCode.py --input data/Lena.png --output encrypted_image --keys K1 K2 K3
```

* `--input_path`: path to the original image
* `--keys`: three secret keys (strings) for encryption



---

### 2. Decrypt an Image

If supported by the base code:

```bash
python src/MultiLayer_BaseCode.py --decrypt --input results/Lena_encrypted.png --output decrypted_image 
```

The decrypted image should match the original image exactly when using the correct keys.

---

### 3. Run Testing Scripts

The repository includes multiple scripts for evaluating encryption robustness:

| Script                 | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `chi_square.py`        | Chi-square test for encrypted image randomness                   |
| `compression.py`       | Evaluate robustness under image compression                      |
| `reconstruction.py`    | Test reconstruction quality after decryption                     |
| `keyVariableLength.py` | Test encryption with keys of different lengths                   |
| `PSNR.py`              | Peak Signal-to-Noise Ratio evaluation                            |
| `dataLoss.py`          | Test robustness against data loss                                |
| `runTestRandomness.py` | Run multiple randomness tests (entropy, correlation, chi-square) |
| `SSMI.py`              | Structural similarity metric evaluation                          |
| `correlation.py`       | Compute horizontal, vertical, and diagonal correlations          |
| `Gaussian.py`          | Gaussian noise robustness testing                                |
| `salt&pepper.py`       | Salt-and-pepper noise robustness testing                         |
| `keySensitivity.py`    | Key sensitivity tests                                            |
| `NPCRandUACI.py`       | NPCR and UACI metrics                                            |
| `dataConvert.py`       | Utilities for data conversion and preprocessing                  |
| `Dynamic.py`           | Dynamic testing utilities for batch experiments                  |



## Notes

* Images must be **grayscale or RGB PNG/JPG**.
* Keys must be **strings of sufficient length** for security.
* All results are **deterministic**: same keys produce the same encrypted images.


---

## Contact

For any questions regarding the code:

* Author: `nazia.2109018@bau.edu.bd`

---

## License

This repository is provided for **research and reproducibility purposes only**. Do not redistribute for commercial use without permission.

```
