import os
import sys
sys.path.append('/path/to/project_directory')
from  src.exception import CustomExection
from  src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"raw.csv")



class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Enther the data ingestion methods or components")

        try:
            df=pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info("Read the data from dataset")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train and test slipt data")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion activite is compeleted")

            return(
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
        
        except Exception as e:
            raise CustomExection(e,sys)
        


if __name__=="__main__":
    logging.info("loggin is stared there")
    obj=DataIngestion()
    obj.initiate_data_ingestion() 