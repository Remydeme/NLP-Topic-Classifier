from easyScraper import EasyScrapper
import pandas as pd


if __name__ == "__main__":
    youtube_urls = pd.read_csv('youtube_urls.csv', delimiter=';')
    for index in range(0, len(youtube_urls)):
        print('_________________ Scraping _________________________________')
        print(f'{youtube_urls.iloc[index]}')
        url = youtube_urls.iloc[index, 0]
        easy_scraper = EasyScrapper(url=url,
                        category=youtube_urls.iloc[index, 1],
                        path=youtube_urls.iloc[index, 2],
                        scrolling_time=youtube_urls.iloc[index, 3])
