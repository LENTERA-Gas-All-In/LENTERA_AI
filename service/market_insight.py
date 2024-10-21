import pandas as pd
import comtradeapicall
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import time
import re
from typing import List
from deep_translator import GoogleTranslator


load_dotenv()

COMTRADE_UN_API_KEY = os.getenv("COMTRADE_UN_API_KEY")

class MarketInsightService():
    def __init__(self) -> None:
        self.chain = self.__setup_chain()
        self.translator =  GoogleTranslator(source="id", target="en")

    def __setup_chain(self):
        chat = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        system = "Kamu adalah analis data yang akan memberikan insight yang dibutuhkan oleh klien"
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        chain = prompt | chat
        return chain

    # Exponential backoff wait function
    def __wait_before_next_request(self, attempt):
        wait_time = 2 ** attempt  # Exponential backoff
        print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
        time.sleep(wait_time)

    # Safe invoke with retry mechanism
    def __safe_invoke(self, chain, input_data):
        retry_attempts = 5
        for attempt in range(retry_attempts):
            try:
                return chain.invoke(input_data)
            except Exception as e:  # Catch general exceptions for rate limit issues
                if "Rate limit" in str(e):  # Basic check for rate limit errors
                    print(f"Rate limit exceeded. Attempt {attempt + 1}")
                    self.__wait_before_next_request(attempt)
                else:
                    raise e
        raise Exception("Failed to process request after multiple attempts")
    
    def __get_hs_code(self, product:str):
        en_product = self.translator.translate(product)
        question=f"What is the HS Code of {en_product}, represent it on [HS CODE] format for the first 6-digit, ex: [151319]"
        response = self.__safe_invoke(self.chain, {"text":question})
        hs_code = re.findall(r'\[(.*?)\]', response.content)
        return hs_code[0].replace(".", "")

    def __get_market_data(self, cmdCode:str, period:str) -> pd.DataFrame:
        market_df = comtradeapicall.getFinalData(COMTRADE_UN_API_KEY,typeCode='C', freqCode='A', clCode='HS', period=period,
                                        reporterCode=None, cmdCode=cmdCode, flowCode='M', partnerCode='360',
                                        partner2Code=None,
                                        customsCode=None, motCode=None, maxRecords=500, format_output='JSON',
                                        aggregateBy=None, breakdownMode='classic', countOnly=None, includeDesc=True)
        return market_df 
    
    def __get_top_n_importer(self, data:pd.DataFrame, n:int=5) -> List[str]:
        group_data = data[["reporterDesc", "netWgt", "primaryValue"]].groupby("reporterDesc").sum().sort_values('netWgt', ascending=False)
        
        if (len(group_data > n)):
            top_n_importer = group_data.iloc[:n].index.to_list()
        else:
            top_n_importer = group_data.index.to_list()
        
        return top_n_importer
    
    def __get_top_n_importer_data(self, data:pd.DataFrame, importer_list:List[str])->dict:
        importer_data={}
        for importer in importer_list:
            importer_data[importer] =  data[data["reporterDesc"]==importer]
        return importer_data

    def __forecast_importer_data(self, data:pd.DataFrame, country:str)->pd.DataFrame:
        important_data = data[["period","reporterDesc", "netWgt", "primaryValue"]]

        question=f"Based on given data. Netwgt is represented in kg, how much {country} will import Coconut oil in kg from Indonesia on 2024 based on weight and primary value? Answer in formatted predicted import: [netWgt; primaryValue], for example: [100; 200]" 
        
        response = self.__safe_invoke(self.chain, {"text": f"Assistant: {question} Data: {important_data.to_dict()}"})

        final_response = self.__safe_invoke(self.chain, {"text":f"Extract netWgt and Primary value prediction from the information in format [netWgt; primaryValue] without any explanation, for example: [100000; 200000].   Information:{response}"})

        matches_response = re.findall(r'\[(.*?)\]', final_response.content)
        forecasted_netWgt, forecasted_primaryValue = tuple(map(float, matches_response[0].replace(",","").split('; ')))
        new_row = {'period':str(int(important_data["period"].iloc[-1])+1), 'reporterDesc':country, 'netWgt':forecasted_netWgt, 'primaryValue':forecasted_primaryValue}

        important_data = important_data._append(new_row, ignore_index=True)

        return important_data
    
    def get_market_insight(self, product: str, period: str = "2023,2022,2021", n:int =5):
        # get HS code
        print("get HS code")
        cmd_code=self.__get_hs_code(product)

        # get market data
        print("get market data")
        market_data = self.__get_market_data(cmd_code, period)

        # get top N product importer with indonesia as the partner
        print("get top N product importer with indonesia as the partner")
        
        top_n_importers = self.__get_top_n_importer(market_data, n)

        # filter top N importer data
        print("filter top N importer data")
        top_n_importers_data  = self.__get_top_n_importer_data(market_data,top_n_importers)

        # forecast netWgt and primary Value for top N importer in the next year
        print("forecast netWgt and primary Value for top N importer in the next year")
        for importer in top_n_importers:
            top_n_importers_data[importer] = self.__forecast_importer_data(top_n_importers_data[importer], importer)

        # return the insight
        print("return the insight")
        return top_n_importers_data