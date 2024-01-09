import csv
import uuid

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# CSV file path
csv_file_path = 'dataset.csv'

# Define CSV fieldnames
fieldnames = ["id", "title", "url", "publisher", "category", "hostname"]

# Set up the web driver
driver = webdriver.Chrome()

# Navigate to the CNN website
driver.get("https://www.nbcnews.com/archive/articles/2023/january")

# Assuming the titles are in h3 tags, adjust this according to the actual HTML structure
main_tag = driver.find_element(By.CLASS_NAME, 'MonthPage')
title_tags = main_tag.find_elements(By.TAG_NAME, 'a')

# Append data to the CSV file
with open(csv_file_path, 'a', newline='') as file:
    titles = []
    for title in title_tags:
        # Generate unique_id
        unique_id = str(uuid.uuid4())
        title_info = {"id": unique_id, "title": title.text, "url": title.get_attribute('href')}
        titles.append(title_info)

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if file.tell() == 0:
        writer.writeheader()

    # Loop through title tags
    for title_info in titles:
        try:
            # Open the URL of each title to extract additional information
            driver.get(title_info.get('url'))

            # Set an implicit wait to wait for elements to be present
            driver.implicitly_wait(10)

            span_tag = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'span[data-testid="unibrow-text"]'))
            )

            div_tag = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-activity-map="inline-byline-article-top"]'))
            )

            # Extract additional information and store it in the title_info dictionary
            title_info['publisher'] = div_tag.text[3:]
            title_info['category'] = span_tag.text
            title_info['hostname'] = 'nbcnews.com'
            print(title_info)

            # Write the data rows
            writer.writerow(title_info)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

print(f"Data has been appended to {csv_file_path}.")

# Close the web driver
driver.quit()
