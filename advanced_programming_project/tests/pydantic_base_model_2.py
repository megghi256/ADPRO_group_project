from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class Airport(BaseModel):
    Name: str
    City: str
    Country: str
    Latitude: float
    Longitude: float
    Airport_ID: str
    IATA: str
    ICAO: str

class Route(BaseModel):
    Airline: str
    Source_airport: str
    Destination_airport: str
    Stops: int
    Equipment: Optional[str] = None

class DatasetConfig(BaseModel):
    data_dir: str
    file_name: str
    processed_data_dir: str
    download_url: HttpUrl
    dataset_names: List[str]
