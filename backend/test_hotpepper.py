import requests
from bs4 import BeautifulSoup

def extract(html):
    soup = BeautifulSoup(html, "html.parser")
    # Finding courses
    # Hotpepper typically uses .courseList .courseItem or similar
    courses = soup.select(".course-list .course-item, .courseList .courseItem, .course-cassette, .courseList li")
    for c in courses:
        name_elem = c.select_one(".course-title, .courseTitle, .course-name, h3, .courseList-name, .courseList-title")
        name = name_elem.get_text(strip=True) if name_elem else ""
        price_elem = c.select_one(".price, .course-price, .courseList-price, .courseList-priceIn")
        price = price_elem.get_text(strip=True) if price_elem else ""
        if name:
            print(f"Course: {name} | Price: {price}")
        
    print(f"Found {len(courses)} courses.")

def main():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    res = requests.get("https://www.hotpepper.jp/strJ001275997/course/", headers=headers)
    if res.status_code == 200:
        extract(res.text)
    else:
        print(f"Failed to fetch {res.status_code}")
        
if __name__ == "__main__":
    main()
