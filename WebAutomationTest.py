import easyocr
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pytesseract
import cv2
from getpass import getpass
import time
import os
import tempfile
from PIL import Image
import io
import base64
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# Set the full path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

def display_captcha(driver, element):
    """Capture and display CAPTCHA image to user"""
    try:
        # Create temp file
        temp_dir = tempfile.mkdtemp()
        captcha_path = os.path.join(temp_dir, "captcha.png")
        
        # Take screenshot of CAPTCHA element
        element.screenshot(captcha_path)
        
        # Display the image
        img = Image.open(captcha_path)
        img.show()
        
        return captcha_path
    except Exception as e:
        print(f"Couldn't display CAPTCHA: {e}")
        return None

def main():
    chromedriver_path = r'C:\Git\Python\chromedriver.exe'

    options = Options()
    # Optional: Keep browser open after script ends
    options.add_experimental_option("detach", True)

    # Set up the driver with your path
    driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
    
    try:
        driver.get("https://nportal.ntut.edu.tw/index.do")
        print("\nNTUT Portal Login\n" + "="*30)

        # Securely get credentials
        username = '113618503'# input("Enter your username: ")
        password = 'Pa$$w0rd'#getpass("Enter your password: ")

        # Fill credentials
        driver.find_element(By.ID, "muid").send_keys(username)
        driver.find_element(By.ID, "mpassword").send_keys(password)

        # CAPTCHA handling
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                captcha_field = driver.find_element(By.ID, "authcode")
                captcha_image = driver.find_element(By.ID, "authImage")
                
                # Display CAPTCHA to user
                print("\nCAPTCHA required - displaying image...")
                captcha_path = display_captcha(driver, captcha_image)

                # Load image
                img = cv2.imread(captcha_path)

                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Invert colors (white on black to black on white)
                gray = cv2.bitwise_not(gray)

                # Resize to help Tesseract
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

                # Denoise
                gray = cv2.medianBlur(gray, 3)

                # Apply threshold
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Load GPU-enabled reader
                reader = easyocr.Reader(['en'], gpu=True)

                # Run OCR
                results = reader.readtext(thresh)

                # Display results
                for (bbox, text, prob) in results:
                    print(f"Detected: {text} (Confidence: {prob:.2f})")

                # # OCR config: Single word, restrict characters
                # config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                # text = pytesseract.image_to_string(thresh, config=config)
                
                cv2.imwrite("processed_thresh.png", thresh)
                Image.open("processed_thresh.png").show()

                if results:
                    text = results[0][1]  # Get the first detected word
                    print("CAPTCHA Text:", text.strip())
                else:
                    text = ""
                    print("No text detected")
                
                if captcha_path:
                    captcha_code = input(f"Attempt {attempt}/{max_attempts} - Enter CAPTCHA: ")
                    os.remove(captcha_path)  # Clean up temp file
                else:
                    captcha_code = input(f"Attempt {attempt}/{max_attempts} - Enter CAPTCHA (check browser): ")
                
                captcha_field.clear()
                captcha_field.send_keys(captcha_code)
                
                # Submit login
                driver.find_element(By.XPATH, "//input[@type='submit']").click()
                time.sleep(2)

                # Check for alerts
                try:
                    alert = Alert(driver)
                    if "驗證碼" in alert.text:
                        print("! CAPTCHA verification failed")
                        alert.accept()
                        continue
                except:
                    pass
                    
                break

            except Exception as e:
                if attempt == max_attempts:
                    print("\n Maximum CAPTCHA attempts reached")
                    raise
                print(f"Retrying... ({e})")

        # Verify login
        time.sleep(3)
        if "authImage" not in driver.page_source and "index.do" not in driver.current_url:
            print("\n Login successful!")
        else:
            print("\n Login failed")

    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
    finally:
        input("\nPress Enter to close the browser...")
        driver.quit()

if __name__ == "__main__":
    main()