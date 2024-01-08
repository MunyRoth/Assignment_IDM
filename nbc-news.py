import csv

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# CSV file path
csv_file_path = 'dataset.csv'

# Check if the file exists, if not create it with header
file_exists = False
last_index = 0

try:
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)  # Convert to a list for easier checking
        if rows:
            header_row = rows[0]
            data_rows = rows[1:] if header_row[0].lower() == 'id' else rows

            if data_rows:
                # Read the last 'id' from the CSV file
                last_index = max(int(row[0]) for row in data_rows)
            else:
                # Handle the case when there are no data rows
                last_index = 0
        else:
            # Handle the case when the file is empty
            last_index = 0
except FileNotFoundError:
    # Handle the case when the file is not found
    last_index = 0

# Set up the web driver (you need to have the appropriate web driver executable installed)
driver = webdriver.Chrome()

# Navigate to the CNN website
driver.get("https://www.nbcnews.com/archive/articles/2023/january")

# Assuming the titles are in h3 tags, adjust this according to the actual HTML structure
main_tag = driver.find_element(By.CLASS_NAME, 'MonthPage')
title_tags = main_tag.find_elements(By.TAG_NAME, 'a')
titles = []

# Generate new 'id' values for titles
index = last_index
for title in title_tags:
    index += 1
    title_info = {"id": index, "title": title.text, "url": title.get_attribute('href')}
    titles.append(title_info)

# Append data to the CSV file
with open(csv_file_path, 'a', newline='') as file:
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

            # Write the title_info dictionary to the CSV file
            fieldnames = title_info.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(title_info)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

print(f"Data has been appended to {csv_file_path}.")

# Close the web driver
driver.quit()
