import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from youtube_transcript_api import YouTubeTranscriptApi as yt
import time

from tqdm import tqdm


class EasyScrapperKeywordError(Exception):

    def __init__(self, message):
        super().__init__(self, message)


class EasyScrapper:
    """
    Scraper Object to scrap youtube videos
    """
    video_metadatas_ =  pd.DataFrame(columns = ['link', 'title', 'description', 'category'])


    # selenium web driver  Chrome object
    driver_ = webdriver.Chrome('./chromedriver')

    # array that store videos links
    video_links_ = []

    # ratio video downloaded
    number_video_transcript_ = 0




    class EasyScrapDecorator:

        def __init__(self):
            pass

        @classmethod
        def time(cls, decorated):
            def wrapper(*args, **kwargs):
                start = time.time()
                rv = decorated(*args, **kwargs)
                end = time.time()
                print(f'{decorated.__name__}, : {end-start} sec')
                return rv
            return wrapper



    def __init__(self, url, category, path,  scrolling_time=10, key_words=None, filtered=False, delim=';', mode='w+'):

        """
        :param url: Youtube url
        :param path: saving path of the csv file
        :param scrolling_time: During how many seconds you want to scroll down
        :param key_words: filter key_words for example ["fruits", "Apple", "Banana"] Will be use to select only element
        :param filter:
        """
        self.url_ = url
        self.category_ = category
        self.pages_ = scrolling_time
        self.filter_ = filtered

        if filter == True: # if filter is equal to False we don't store the keyword
            if key_words != None:
                self.key_words_ = key_words
            else:
                raise Exception("The parameter keyword is absent. key_words must be set if filter is equal to true")
        self.fetch_links()
        self.fetch_videos_metadatas()
        print(self.video_metadatas_.head())
        self.fetch_video_transcriptions()
        self.video_metadatas_.head()
        self.to_csv(path=path,delim=delim, mode=mode)


    @EasyScrapDecorator.time
    def __scroll_down(self):

        """
        This function simulate a scrolling action. On each page there is 20 videos display. If we want to dsiplay
        2000 videos we need to scroll 100.
        :param pages: During how many pages we display
        :return:
        """
        SCROLL_PAUSE_TIME = 1
        pages_counter = 0
        last_value = 0

        try:
            # Get scroll height
            while self.pages_ > pages_counter:
                time.sleep(4)
                scroll_height = self.driver_.execute_script("return document.documentElement.scrollHeight")
                if last_value == scroll_height:#hit the bottom of the page
                    break
                self.driver_.find_element_by_tag_name('body').send_keys(Keys.END)
                pages_counter += SCROLL_PAUSE_TIME
                last_value = scroll_height
                print(f"Scroll Height {scroll_height}")
        except Exception:
            print("Hit the bottom of page")
            return

    def fetch_links(self):

        """
        Function is called to fetch all the video metadata online. It use the url and scrap online.
        Once fetch video's metadatas are stored in a Dataframe.
        :return:
        """

        self.driver_.get(self.url_)  # EasyMovie youtube channel

        self.__scroll_down()

        videos_data = self.driver_.find_elements_by_xpath('//*[@id="video-title"]')  # video title XPath
        for index in tqdm(range(0, len(videos_data)-1)):
            link = videos_data[index].get_attribute('href')
            if link == None:
                continue
            self.video_links_.append(link)

        print(f"{len(self.video_links_)} video links have been found.")


    def fetch_videos_metadatas(self):
        """
        Use selenium scrape a youtube web page, fetch all the video links, title, description on this page. A pandas
        dataframe is created with them.
        :return:
        """
        links = []
        titles = []
        descriptions = []
        categories = []

        wait = WebDriverWait(self.driver_, 10) # pause

        for index in tqdm(range(0, len(self.video_links_))):
            print("Index value", index, "Links : ", self.video_links_[index])
            link = self.video_links_[index]
            print("Display link", link)
            self.driver_.get(link)
            v_id = link.strip('https://www.youtube.com/watch?v=')
            v_title = wait.until(EC.presence_of_element_located(
                       (By.CSS_SELECTOR, "h1.title yt-formatted-string"))).text
            v_description = wait.until(EC.presence_of_element_located(
                                     (By.CSS_SELECTOR, "div#description yt-formatted-string"))).text
            links.append(v_id)
            titles.append(v_title)
            descriptions.append(v_description)
            categories.append(self.category_)
        self.video_metadatas_ = pd.DataFrame({'link': links, 'title': titles, 'description': descriptions, 'category': categories})


        if self.filter_ == True:
            self.__filter_videos()


    @EasyScrapDecorator.time
    def __filter_videos(self):

        """
        Remove from the dataframe all the row that doesn't contains one of the word specified in the keywords list
        """
        dataframes = []
        for word in self.key_words_:
            df = self.video_metadatas_[self.video_metadatas_['title'].str.contains(word)]
            dataframes.append(df)
        concat_dataframe = pd.concat(dataframes)
        concat_dataframe = concat_dataframe['title'].unique()
        self.video_metadatas_ = concat_dataframe

    @EasyScrapDecorator.time
    def __clean_text(self, transcripts=None):
        """"
            Metod use to clean the transcription text. It remove all the time and bad keywords
        """
        clean_transcripts = []
        if transcripts != None:
            for index in tqdm(range(0, len(transcripts))):
                clean_transcripts.append(' ')
                if transcripts[index] == None: # if there is no text transcription
                    continue
                for _, text_array in transcripts[index][0].items():
                    for dico in text_array:
                            clean_transcripts[index] += ' ' + dico['text']
        return clean_transcripts


    @EasyScrapDecorator.time
    def fetch_video_transcriptions(self):
        """
            Use videos link and selenium to download the video transcription
        """
        transcripts = []
        #iter_on_row = ((index, row) for index, row in self.video_metadatas_.iterrows())
        for index in tqdm(range(0, len(self.video_metadatas_))):
            row = self.video_metadatas_.iloc[index]
       #         if index % 100:
       #        time.sleep(60) # take a pause of 1 minute after 100 translation api call to avoid
                               # if not google may stop the bot
            try:
                transcript = yt.get_transcripts({row['link']}, languages=['fr'])
                transcripts.append(transcript)
                self.number_video_transcript_ += 1
                #if index % 10:
                 #   self.save_on_running(transcripts)
            except Exception:
                transcripts.append(None)
        transcripts = self.__clean_text(transcripts=transcripts)
        self.video_metadatas_['transcriptions'] = transcripts

    @EasyScrapDecorator.time
    def to_csv(self, path, delim=";", mode='w+'):
        """
            Videos Metadata and transcriptions are load in a csv.
        """
        self.video_metadatas_.to_csv(path, index=True, header=True, sep=delim, encoding='utf-8',mode=mode)

    def save_on_running(self, transcripts, delim=";"):
        """
        Save on running
        """
        sub_df = self.video_metadas_.iloc[len(transcripts), :]
        sub_df['transcriptions'] = transcripts
        self.video_metadas_.to_csv(path, index=True, header=True, sep=delim, encoding='utf-8', mode="w+")


if __name__ == "__main__":
    easy_scraper = EasyScrapper(url='https://www.youtube.com/results?search_query=témoignage+client&sp=EgIYAQ%253D%253D',
                                category='temoignage',
                                path='./dataset/temoignage_client_data.csv',
                                scrolling_time=400)
