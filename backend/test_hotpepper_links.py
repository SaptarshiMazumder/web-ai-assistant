import requests
from bs4 import BeautifulSoup

def main():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    # Using a valid Hotpepper URL I found in platform_profiles placeholder: J001234567 
    # Or let's use a known big restaurant: strJ000000000 (often not found). Let's use the one from the config: strJ001234567
    
    # Actually, J001234567 might also be a dummy. Let's hit the root domain and search or just see what an arbitrary one looks like.
    # A real one is strJ000028292 or strJ001000570
    res = requests.get("https://www.hotpepper.jp/strJ001000570/", headers=headers)
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, "html.parser")
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/strJ'):
                # Extract the path part
                parts = href.split('/')
                if len(parts) > 2:
                    subpath = '/' + parts[2]
                    links.add(subpath)
        for link in sorted(list(links)):
            print(link)
    else:
        print(f"Failed to fetch {res.status_code}")
        
if __name__ == "__main__":
    main()
