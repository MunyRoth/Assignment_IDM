import csv
import time
import uuid

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# CSV file path
csv_file_path = 'dataset.csv'

# Define CSV fieldnames
fieldnames = ["id", "title", "url", "publisher", "category", "hostname"]

# Set up the web driver
driver = webdriver.Chrome()

# Navigate to the Yahoo website
driver.get("https://www.yahoo.com/news")

# Scroll down multiple times to load more content
for _ in range(5):  # Adjust the number of times you want to scroll
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    time.sleep(2)  # Wait for the content to load (you can adjust the waiting time if needed)

title_tags = driver.find_elements(By.CSS_SELECTOR, 'h3[data-test-locator="stream-item-title"]')

# Append data to the CSV file
with open(csv_file_path, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if file.tell() == 0:
        writer.writeheader()

    # Loop title_tags
    for title in title_tags:
        try:
            a_tag = title.find_element(By.TAG_NAME, 'a')
            url = a_tag.get_attribute('href')
            title_text = a_tag.text

            previous_sibling = title.find_element(By.XPATH, 'preceding-sibling::*')
            category_tag = previous_sibling.find_element(By.TAG_NAME, 'strong')
            publisher_tag = previous_sibling.find_element(By.CSS_SELECTOR,
                                                          'span[data-test-locator="stream-item-publisher"]')

            # Extract additional information and store it in the title_info dictionary
            publisher = publisher_tag.text
            category = category_tag.text
            hostname = 'yahoo.com'

            # Generate unique_id
            unique_id = str(uuid.uuid4())

            # Write the title_info dictionary to the CSV file
            title_info = {
                "id": unique_id,
                "title": title_text,
                "url": url,
                "publisher": publisher,
                "category": category,
                "hostname": hostname
            }

            # Write the data rows
            writer.writerow(title_info)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

print(f"Data has been appended to {csv_file_path}.")

# Close the web driver
driver.quit()
