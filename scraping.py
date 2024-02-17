# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # Initialize the Chrome webdriver
# driver = webdriver.Chrome()

# # Base URL of the webpage to scrape
# base_url = "https://www.zomato.com/mumbai/ettarra-1-juhu/reviews?page={}&sort=dd&filter=reviews-dd"

# # Function to scrape data
# def scrape_data(url):
#     driver.get(url)
    
#     # Find all elements with class name "bcCauD" for names
#     names = driver.find_elements(By.CLASS_NAME, "bcCauD")
#     for name in names:
#         print("Name:", name.text)

#     # Find all elements with class name "dJxGwQ" for reviews
#     reviews = driver.find_elements(By.CLASS_NAME, "dJxGwQ")
#     for review in reviews:
#         print("Review:", review.text)

#     # Find all elements with class name "dYrjiw" for order type
#     orders = driver.find_elements(By.CLASS_NAME, "dYrjiw")
#     for order in orders:
#         print("Order Type:", order.text)

#     # Find all elements with class name "time-stamp" for date
#     dates = driver.find_elements(By.CLASS_NAME, "time-stamp")
#     for date in dates:
#         print("Date:", date.text)

# # Start iterating through the links
# page_number = 1
# while True:
#     url = base_url.format(page_number)
#     driver.get(url)

#     # Check if the page exists
#     try:
#         WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, "bcCauD")))
#         scrape_data(url)
#         page_number += 1
#     except:
#         break

# # Close the webdriver
# driver.quit()


import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize the Chrome webdriver
driver = webdriver.Chrome()

# Base URL of the webpage to scrape
base_url = "https://www.zomato.com/mumbai/ettarra-1-juhu/reviews?page={}&sort=dd&filter=reviews-dd"

# Function to scrape data
def scrape_data(url):
    driver.get(url)
    
    # Find all elements with class name "bcCauD" for names
    names = driver.find_elements(By.CLASS_NAME, "bcCauD")
    
    # Find all elements with class name "dJxGwQ" for reviews
    reviews = driver.find_elements(By.CLASS_NAME, "dJxGwQ")
    
    # Find all elements with class name "dYrjiw" for order type
    orders = driver.find_elements(By.CLASS_NAME, "dYrjiw")
    
    # Find all elements with class name "time-stamp" for date
    dates = driver.find_elements(By.CLASS_NAME, "time-stamp")
    
    # Open the CSV file in write mode with newline='' to prevent extra newline characters
    with open('C:/Users/HARGUN/Desktop/hackniche/scrape_data.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write headers if the file is empty
        if file.tell() == 0:
            writer.writerow(['Name', 'Review', 'Order Type', 'Date'])
        
        # Iterate through the scraped data and write to CSV
        for name, review, order, date in zip(names, reviews, orders, dates):
            writer.writerow([name.text, review.text, order.text, date.text])

# Start iterating through the links
page_number = 1
while True:
    url = base_url.format(page_number)
    driver.get(url)

    # Check if the page exists
    try:
        WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.CLASS_NAME, "bcCauD")))
        scrape_data(url)
        page_number += 1
    except:
        break

# Close the webdriver
driver.quit()
