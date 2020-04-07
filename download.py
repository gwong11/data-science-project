import csv
import shutil
import os
import json
from urllib.request import urlopen
from collections import defaultdict

'''
This module retrieves files from a REST API
'''
class DownloadFiles:

    '''
    Initialize the class by providing a URL to download files 
    with an optional filter date parameter
    '''
    def __init__(self, base_url):

        self.base_url = base_url
        self.res_format = 'json'
        self.papers = defaultdict(list)
        
    '''
    Method to do a GET to retrieve the response, given the URL
    '''
    def retrieve(self, pmid):
        paper = None
        found = True
        url = self.base_url + self.res_format + '/' + pmid + '/unicode'
        try:
            with urlopen(url) as response:
                paper = response.read().decode('utf-8')

            print(pmid + " downloaded!")
            self.papers['papers'].append(json.loads(paper))
        except:
            found = False
            print(pmid + " doesn't exist!")

        return found

    '''
    Getter to retrieve the papers dict
    '''
    def getPapersDict(self):
        return self.papers

    '''
    Clear the papers dict
    '''
    def clearPapersDict(self):
        self.papers = defaultdict(list)

    '''
    Method to write JSON string to disk
    '''
    def write(self, json_data, directory, filename):
        with open(os.path.join(directory, filename), 'w') as outfile:
            json.dump(json_data, outfile)

    '''
    Method to pretty print the JSON
    '''
    def pprint_JSON(self, json_string):
        parsed = json.loads(json_string)
        print(json.dumps(parsed, indent=4, sort_keys=True))
